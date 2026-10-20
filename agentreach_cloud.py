"""Use upstream Agent-Reach's YouTube/Web channels in Streamlit Cloud.

Intentionally does not read cookies or execute arbitrary user shell commands.
"""
import html
import json
import re
from urllib.parse import urlsplit

import requests


def _allowed(host, domains):
    host = (host or "").lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in domains)


def check_channels():
    from agent_reach.channels.web import WebChannel
    from agent_reach.channels.youtube import YouTubeChannel
    result = {}
    for channel in [WebChannel(), YouTubeChannel()]:
        status, info = channel.check()
        result[channel.name] = (status, channel.active_backend or "none", info[:240])
    return result


def read_guide(url):
    """Use upstream Agent-Reach WebChannel.read (Jina Reader)."""
    from agent_reach.channels.web import WebChannel
    p = urlsplit(url.strip())
    if (p.scheme != "https" or p.username or p.password or p.port not in (None, 443)
            or not _allowed(p.hostname, ("tvmaze.com", "wikipedia.org", "fandom.com", "thetvdb.com"))):
        raise ValueError("Guide must be an HTTPS URL at TVMaze, Wikipedia, Fandom or TheTVDB.")
    return WebChannel().read(url.strip())[:6000]


def _captions(text, ext):
    if ext == "json3":
        events = json.loads(text).get("events", [])
        return re.sub(r"\s+", " ", "".join(s.get("utf8", "") for e in events for s in e.get("segs", []))).strip()[:8500]
    rows = []
    for row in text.splitlines():
        row = row.strip()
        if not row or row.isdigit() or row.startswith(("WEBVTT", "Kind:", "Language:")) or "-->" in row:
            continue
        row = html.unescape(re.sub(r"<[^>]+>", "", row))
        if not rows or row != rows[-1]:
            rows.append(row)
    return " ".join(rows)[:8500]


def read_youtube_captions(url):
    """Use yt-dlp, the backend selected by upstream Agent-Reach YouTubeChannel."""
    from agent_reach.channels.youtube import YouTubeChannel
    from reachlab import normalize_url, is_youtube
    from yt_dlp import YoutubeDL
    url = normalize_url(url)
    if not is_youtube(url) or not YouTubeChannel().can_handle(url):
        raise ValueError("Provide a public YouTube video/Short URL.")
    with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True,
                    "noplaylist": True, "socket_timeout": 8, "retries": 0,
                    "ignore_no_formats_error": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    if not isinstance(info, dict) or info.get("_type") in ("playlist", "multi_video"):
        raise ValueError("Single public video could not be read.")
    for kind in ("subtitles", "automatic_captions"):
        tracks = info.get(kind) or {}
        for lang in sorted(tracks, key=lambda x: (not x.startswith("en"), x)):
            for ext in ("json3", "vtt"):
                candidate = next((x for x in tracks[lang] if x.get("ext") == ext and x.get("url")), None)
                if not candidate:
                    continue
                u = candidate["url"]
                p = urlsplit(u)
                if p.scheme != "https" or not _allowed(p.hostname, ("youtube.com", "googlevideo.com", "google.com")):
                    continue
                try:
                    with requests.get(u, timeout=10, stream=True) as response:
                        response.raise_for_status()
                        parts, size = [], 0
                        for block in response.iter_content(8192):
                            size += len(block)
                            if size > 250000:
                                raise ValueError("Caption payload too large")
                            parts.append(block)
                    result = _captions(b"".join(parts).decode("utf-8", "replace"), ext)
                    if len(result) > 20:
                        return result, f"yt-dlp {kind} ({lang})"
                except (ValueError, requests.RequestException, json.JSONDecodeError):
                    continue
    raise RuntimeError("No accessible captions found. Add a screenshot or dialogue notes.")
