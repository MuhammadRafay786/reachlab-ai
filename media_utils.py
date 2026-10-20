"""Bounded user-uploaded video sampling and best-effort YouTube captions."""
from __future__ import annotations

import html
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import requests

MAX_VIDEO_BYTES = 45 * 1024 * 1024
ALLOWED_EXTENSIONS = {"mp4", "mov", "webm", "m4v"}


def extract_video_frames_and_audio(data: bytes, extension: str,
                                   want_audio: bool = False) -> tuple[list[bytes], bytes | None]:
    """Sample max eight low-resolution JPEG frames from 90 seconds of an uploaded video.

    Do not download any remote video. Inputs are user-provided bytes; use temporary
    directories and bounded subprocess timeouts. Extra API spend for audio is opt-in.
    """
    import imageio_ffmpeg

    extension = extension.lower().lstrip(".")
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Allowed uploaded video types: MP4, MOV, WEBM, M4V")
    if not data or len(data) > MAX_VIDEO_BYTES:
        raise ValueError("Video must be under 45 MB")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix="reachlab_upload_") as tmp:
        inp = Path(tmp) / f"input.{extension}"
        inp.write_bytes(data)
        pattern = str(Path(tmp) / "frame_%02d.jpg")
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-i", str(inp), "-t", "90", "-vf", "fps=1/8,scale=640:-2",
               "-frames:v", "8", "-q:v", "5", pattern]
        run = subprocess.run(cmd, capture_output=True, timeout=35, check=False)
        frames = [p.read_bytes() for p in sorted(Path(tmp).glob("frame_*.jpg"))]
        if not frames:
            # Very short clips can produce zero frames at one-per-eight-seconds.
            first = Path(tmp) / "first_frame.jpg"
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                            "-i", str(inp), "-frames:v", "1", "-q:v", "5", str(first)],
                           capture_output=True, timeout=20, check=False)
            if first.exists():
                frames = [first.read_bytes()]
        if not frames:
            raise ValueError("Could not decode this video; please upload MP4 or screenshots.")
        audio = None
        if want_audio:
            mp3 = Path(tmp) / "audio.mp3"
            arun = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                 "-i", str(inp), "-t", "90", "-vn", "-ac", "1", "-ar", "16000",
                 "-b:a", "48k", str(mp3)],
                capture_output=True, timeout=35, check=False,
            )
            if arun.returncode == 0 and mp3.exists():
                audio = mp3.read_bytes()
        return frames, audio


def transcribe_audio(api_key: str, mp3: bytes, client=None) -> str:
    """Optional separate transcription request; extra usage-based charge."""
    from io import BytesIO
    from openai import OpenAI

    client = client or OpenAI(api_key=api_key, timeout=60.0, max_retries=1)
    file = BytesIO(mp3)
    file.name = "scene.mp3"
    reply = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=file)
    return str(getattr(reply, "text", "") or "")[:9000]


def caption_text(url: str, maximum_chars: int = 9500, info: dict | None = None) -> str:
    """Read a bounded publicly accessible YouTube caption track when available.

    No browser cookies. Cloud IPs/YouTube may block these. User's link is validated
    externally; this helper only sends caption requests to known caption hosts.
    """
    from yt_dlp import YoutubeDL
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "noplaylist": True, "socket_timeout": 8, "retries": 0,
            "cachedir": False, "ignore_no_formats_error": True}
    if info is None:
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception:
            return ""
    if not isinstance(info, dict):
        return ""
    tracks = info.get("subtitles") or info.get("automatic_captions") or {}
    if not isinstance(tracks, dict):
        return ""
    langs = list(tracks)
    ordered = [x for x in ("en", "en-US", "en-orig") if x in tracks]
    ordered += [x for x in langs if x not in ordered][:2]
    for lang in ordered[:4]:
        choices = tracks.get(lang) or []
        for track in sorted(choices, key=lambda t: t.get("ext") != "vtt"):
            if track.get("ext") not in {"vtt", "srv3", "ttml"}:
                continue
            caption_url = track.get("url", "")
            try:
                host = (urlsplit(caption_url).hostname or "").lower()
                if not (host == "youtube.com" or host.endswith(".youtube.com") or
                        host == "googlevideo.com" or host.endswith(".googlevideo.com")):
                    continue
                with requests.get(caption_url, timeout=8, stream=True, allow_redirects=False) as res:
                    res.raise_for_status()
                    chunks, total = [], 0
                    for chunk in res.iter_content(8192):
                        total += len(chunk)
                        if total > 400_000:
                            break
                        chunks.append(chunk)
                raw = b"".join(chunks).decode("utf-8", errors="replace")
                lines = []
                for line in raw.splitlines():
                    line = line.strip()
                    if not line or line.startswith(("WEBVTT", "NOTE", "Kind:", "Language:")):
                        continue
                    if re.search(r"\d\d:\d\d:\d\d", line) or line.isdigit():
                        continue
                    clean = re.sub(r"<[^>]+>", "", line)
                    clean = html.unescape(clean)
                    if clean and (not lines or clean != lines[-1]):
                        lines.append(clean)
                if lines:
                    return " ".join(lines)[:maximum_chars]
            except requests.RequestException:
                continue
    return ""


def youtube_public_evidence(url: str) -> tuple[dict, str]:
    """Read metadata + available captions with ONE yt-dlp extract (faster)."""
    from yt_dlp import YoutubeDL
    from reachlab import safe_string

    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "noplaylist": True, "socket_timeout": 8, "retries": 0,
            "cachedir": False, "ignore_no_formats_error": True}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not isinstance(info, dict) or info.get("_type") in ("playlist", "multi_video"):
        raise ValueError("Single public YouTube video required")
    meta = {"title": safe_string(info.get("title"), 300),
            "creator": safe_string(info.get("uploader") or info.get("channel"), 150),
            "description": safe_string(info.get("description"), 2400)}
    return meta, caption_text(url, info=info)
