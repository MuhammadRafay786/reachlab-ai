"""ReachLab AI — Cloud-ready Episode Finder; OpenAI fast + verify + Gemini fallback."""
from __future__ import annotations

import hashlib
import hmac
import io
import os
import time
from urllib.parse import quote

import streamlit as st

from media_utils import extract_video_frames_and_audio, transcribe_audio, youtube_public_evidence
from openai_backend import FAST_MODEL, VERIFY_MODEL, analyze_openai, research_openai
from reachlab import (
    episode_match_status, format_download, gemini_analyze, grounded_web_research,
    is_youtube, normalize_url, public_media_metadata, safe_string, tvmaze_verify,
)

st.set_page_config(page_title="ReachLab AI | Episode Finder", page_icon="🎬", layout="centered")
st.markdown("""<style>
.block-container {max-width: 930px; padding-top: 1.75rem; padding-bottom: 4rem;}
[data-testid="stApp"] {background: radial-gradient(ellipse at top left,rgba(94,82,255,.085),transparent 46%);}
.kicker {font-size:.72rem;letter-spacing:.17em;font-weight:800;color:#a7a1ff;}
.hero {padding:1.25rem 0 1.15rem;}
.hero h1 {font-size:2.9rem;line-height:1.15;letter-spacing:-.055em;margin:.5rem 0;}
.hero p {font-size:1rem;color:#9ca0b2;}
</style>""", unsafe_allow_html=True)


def secret(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:
        return ""


def cache_key(*values: object) -> str:
    hasher = hashlib.sha256()
    for value in values:
        part = value if isinstance(value, bytes) else str(value).encode("utf-8")
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return hasher.hexdigest()


def screenshot_contact_sheet(frames: list[bytes]) -> bytes:
    """Use a small labelled 2-column collage as Gemini's screenshot fallback."""
    from PIL import Image, ImageDraw
    images = []
    for blob in frames[:8]:
        with Image.open(io.BytesIO(blob)) as im:
            im = im.convert("RGB")
            im.thumbnail((480, 270))
            images.append(im.copy())
    if not images:
        return b""
    canvas = Image.new("RGB", (960, ((len(images) + 1) // 2) * 290), "#15171f")
    pen = ImageDraw.Draw(canvas)
    for i, im in enumerate(images):
        x = (i % 2) * 480
        y = (i // 2) * 290
        canvas.paste(im, (x, y + 18))
        pen.text((x + 8, y + 2), f"Sampled frame {i + 1}", fill="#ffffff")
    out = io.BytesIO()
    canvas.save(out, "JPEG", quality=75)
    return out.getvalue()


password = secret("APP_PASSWORD")
openai_key = secret("OPENAI_API_KEY")
gemini_key = secret("GEMINI_API_KEY")
openai_fast_model = secret("OPENAI_FAST_MODEL") or FAST_MODEL
openai_verify_model = secret("OPENAI_VERIFY_MODEL") or VERIFY_MODEL
gemini_model = secret("GEMINI_MODEL") or "gemini-3.8-flash"

if len(password) < 12:
    st.error("Set APP_PASSWORD (12+ characters) in Streamlit App settings → Secrets before continuing.")
    st.stop()

if not st.session_state.get("authenticated"):
    st.markdown('<div class="hero"><div class="kicker">PRIVATE RESEARCH WORKSPACE</div><h1>ReachLab<span style="color:#a8a5ff">.</span></h1><p>Find the original episode behind a short clip.</p></div>', unsafe_allow_html=True)
    entry = st.text_input("Access password", type="password", key="access_password")
    if st.button("Unlock", type="primary", use_container_width=True):
        attempts = st.session_state.get("failed_logins", 0)
        if attempts >= 8:
            st.error("Too many attempts in this session. Try again later.")
        elif hmac.compare_digest(entry, password):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.session_state.failed_logins = attempts + 1
            st.error("Incorrect password")
    st.stop()

st.markdown('<div class="hero"><div class="kicker">VIDEO & EPISODE RESEARCH</div><h1>Find the scene. <span style="color:#aaa4ff">Find the episode.</span></h1><p>Paste a public link, add a clip or screenshot, and investigate the original TV/cartoon episode with transparent evidence.</p></div>', unsafe_allow_html=True)

if not openai_key and not gemini_key:
    st.error("No AI provider configured. Add OPENAI_API_KEY or GEMINI_API_KEY in Streamlit Secrets.")
    st.stop()

with st.sidebar:
    st.header("Research settings")
    primary = st.radio("AI provider", ["OpenAI (fast)" , "Gemini" ] if openai_key and gemini_key else (["OpenAI (fast)"] if openai_key else ["Gemini"]), index=0)
    use_fallback = st.checkbox("Automatically try Gemini if OpenAI fails", value=bool(gemini_key), disabled=not bool(gemini_key) or not bool(openai_key))
    language = st.selectbox("Language", ["Roman Urdu", "English", "Urdu"])
    goal = st.selectbox("Goal", ["Identify the original TV/cartoon episode", "Summarize scene for clipping channel", "Write captions and editing notes"])
    catalog = st.checkbox("Check TVMaze episode listing", value=True)
    st.divider()
    st.caption("OpenAI requests, web search and optional transcription are usage-billed. Gemini fallback may also be billed. Use account spending limits.")
    st.caption("Agent-Reach is not fully integrated. This MVP uses metadata, captions, uploaded video, and authorized AI APIs. No social-login cookies are stored.")
    st.caption("Use clips legally; episode identification does not grant publishing rights.")

# Explicit diagnostic: never show or log the actual API key.
with st.sidebar:
    st.divider()
    if openai_key and st.button("Test OpenAI connection (small API charge)", use_container_width=True):
        try:
            from openai import OpenAI
            test_client = OpenAI(api_key=openai_key, timeout=20.0, max_retries=0)
            test_response = test_client.responses.create(
                model=openai_fast_model, input="Reply with the word OK.",
                max_output_tokens=16, store=False,
            )
            if (getattr(test_response, "output_text", "") or "").strip():
                st.success(f"OpenAI connection works: {openai_fast_model}")
            else:
                st.warning("OpenAI request succeeded but returned no text. Check model response details.")
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            kind = type(exc).__name__
            # Exception text can contain sensitive details; display only safe metadata.
            st.error(f"OpenAI test failed: {kind}" + (f" (HTTP {status})" if status else ""))
            if status == 401:
                st.info("Check whether the OpenAI API key is valid and active.")
            elif status == 429:
                st.info("Check OpenAI API credits, quota and rate limits.")
            elif status == 403:
                st.info("Check project permissions and model access.")
            elif status == 400:
                st.info("Check OPENAI_FAST_MODEL and request compatibility.")
            else:
                st.info("Open Manage app → Logs to inspect the error category (never post keys).")
    if gemini_key and st.button("Test Gemini connection (small API charge)", use_container_width=True):
        try:
            from google import genai
            gemini_client = genai.Client(api_key=gemini_key)
            gemini_test = gemini_client.models.generate_content(
                model=gemini_model, contents="Respond with the word OK."
            )
            if (getattr(gemini_test, "text", "") or "").strip():
                st.success(f"Gemini connection works: {gemini_model}")
            else:
                st.warning("Gemini returned no text. Check model access or safety filtering.")
        except Exception as exc:
            status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            st.error(f"Gemini test failed: {type(exc).__name__}" + (f" (status {status})" if status is not None else ""))
            st.info("Check GEMINI_MODEL, API project/model access, quota and billing. The API key is not displayed.")
    if st.button("Check Agent-Reach channels", use_container_width=True):
        try:
            from agentreach_cloud import check_channels
            channel_status = check_channels()
            for channel, (health, backend, _description) in channel_status.items():
                st.write(f"{channel}: {health} (backend: {backend})")
            st.caption("Status checks tools only; they do not generate an AI answer.")
        except Exception as exc:
            st.error(f"Agent-Reach check failed: {type(exc).__name__} (see private Streamlit logs)")
    if st.button("Clear saved analysis and retry", use_container_width=True):
        for field in ("analysis_cache", "current_result", "deep_result", "last_started"):
            st.session_state.pop(field, None)
        st.rerun()

url = st.text_input("YouTube Short / Instagram / Reddit link", placeholder="https://youtube.com/shorts/...")
notes = st.text_area("Optional scene description or exact dialogue", placeholder="What happened? Any dialogue or character names?", max_chars=10000, height=90)
known_season = st.number_input("Known season (optional clue, 0 = unknown)", min_value=0, max_value=100, value=0)
img = st.file_uploader("Optional scene screenshot", type=["jpg", "jpeg", "png"], max_upload_size=7)
video = st.file_uploader("Optional short video upload — MP4 / MOV / WEBM (max 45 MB)", type=["mp4", "mov", "webm", "m4v"], max_upload_size=45)
want_transcript = st.checkbox("Transcribe uploaded video's spoken audio (separate API charge)", value=False, disabled=not bool(openai_key))
st.caption("For link-only YouTube, ReachLab tries public captions + metadata. To analyze visuals reliably, upload the short video or a screenshot; a URL alone doesn't supply frames to OpenAI.")

analyze = st.button("Analyze clip — Fast", type="primary", use_container_width=True)

if analyze:
    if time.monotonic() - st.session_state.get("last_started", -10**10) < 8:
        st.warning("Please wait a few seconds between requests.")
        st.stop()
    st.session_state.last_started = time.monotonic()
    try:
        clean_url = normalize_url(url) if url.strip() else None
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if not any([clean_url, notes.strip(), img, video]):
        st.error("Add a link, scene notes, screenshot, or uploaded video.")
        st.stop()
    image_data = img.getvalue() if img else None
    image_mime = "image/png" if img and img.type == "image/png" else "image/jpeg"
    video_data = video.getvalue() if video else None
    key = cache_key(clean_url, notes, known_season, language, goal, primary, openai_fast_model, gemini_model, catalog,
                    image_data or b"", video_data or b"", bool(want_transcript))
    cached = st.session_state.get("analysis_cache", {})
    if key in cached:
        st.info("Loaded your previous analysis from this session (no new AI call).")
        st.session_state["current_result"] = cached[key]
    else:
        evidence = []
        added_notes = notes.strip()
        frames, extracted_audio = [], None
        if video_data:
            try:
                with st.spinner("Sampling short video frames (local processing)..."):
                    frames, extracted_audio = extract_video_frames_and_audio(
                        video_data, video.name.rsplit(".", 1)[-1],
                        want_audio=bool(want_transcript and openai_key))
                evidence.append(f"{len(frames)} sampled video frames")
            except Exception as exc:
                st.error("Uploaded video could not be decoded: " + safe_string(exc, 250))
                st.stop()
        if extracted_audio and openai_key:
            try:
                with st.spinner("Transcribing spoken audio (extra API request)..."):
                    transcript = transcribe_audio(openai_key, extracted_audio)
                if transcript.strip():
                    added_notes += "\nUPLOAD AUDIO TRANSCRIPT (machine-generated; may have errors):\n" + transcript
                    evidence.append("audio transcript")
            except Exception as exc:
                st.warning("Audio transcription unavailable: " + safe_string(type(exc).__name__, 50) + ". Visual analysis will continue.")
        if clean_url and not video_data:
            # One public YouTube fetch for metadata + caption tracks; no video downloaded.
            try:
                with st.spinner("Reading available public captions and metadata..."):
                    if is_youtube(clean_url):
                        m, captions = youtube_public_evidence(clean_url)
                    else:
                        m, captions = public_media_metadata(clean_url), ""
                if captions:
                    added_notes += "\nYOUTUBE CAPTIONS (automatic captions may be inaccurate):\n" + captions
                    evidence.append("public captions")
                    st.session_state["reachlab_last_main_caption"] = (clean_url, captions)
                    with st.expander("Public captions retrieved (usable even if AI fails)", expanded=False):
                        st.text_area("Extracted transcript", captions, height=160, disabled=True)
                added_notes += ("\nPUBLIC METADATA (NOT scene verification):\n"
                                + f"Title: {m['title']}\nCreator: {m['creator']}\n"
                                + f"Description: {m['description'][:2400]}")
                evidence.append("public metadata")
            except Exception:
                st.info("The platform did not provide public metadata; proceeding with uploaded evidence / notes if available.")

        if not added_notes.strip() and not frames and not image_data and primary.startswith("OpenAI"):
            st.error("OpenAI cannot identify the episode from an inaccessible URL alone. Please upload a scene screenshot/video or enter distinctive dialogue.")
            st.stop()

        result, evidence_label, used_provider = None, "", ""
        err = None
        if primary.startswith("OpenAI") and openai_key:
            try:
                with st.spinner("OpenAI analyzing scene evidence..."):
                    result, evidence_label = analyze_openai(
                        openai_key, openai_fast_model, clean_url, added_notes,
                        language, goal, image_data, image_mime, frames,
                        int(known_season) if known_season else None,
                    )
                used_provider = "OpenAI"
            except Exception as exc:
                err = exc
                if not use_fallback or not gemini_key:
                    st.error("OpenAI analysis failed: " + safe_string(exc, 420))
                    st.stop()
                st.warning(
                    f"OpenAI failed: {type(exc).__name__}"
                    + (f" (HTTP {getattr(exc, 'status_code')})" if getattr(exc, "status_code", None) else "")
                    + ". Trying Gemini fallback. Use 'Test OpenAI connection' for diagnosis."
                )
        if result is None and gemini_key:
            try:
                shot = screenshot_contact_sheet(frames) if frames else image_data
                mime = "image/jpeg" if frames else image_mime
                # For uploaded frames prefer attached scene evidence over remote URL handling.
                fallback_url = None if frames else clean_url
                with st.spinner("Analyzing with Gemini..."):
                    result, evidence_label = gemini_analyze(
                        gemini_key, gemini_model, fallback_url,
                        added_notes, shot, mime, language, goal,
                        prefer_text_evidence=bool(shot or "YOUTUBE CAPTIONS" in added_notes),
                    )
                used_provider = "Gemini" + (" fallback" if err else "")
            except Exception as exc:
                st.error("Gemini request failed: " + safe_string(exc, 400))
                st.stop()
        if result is None:
            st.error("No AI provider produced a result.")
            st.stop()
        with st.spinner("Checking episode catalog (does not verify clip origin)..."):
            listing = tvmaze_verify(safe_string(result.get("show"), 120), result.get("season"),
                                    result.get("episode"), safe_string(result.get("episode_title"), 160)) if catalog else {
                                        "status": "unverified", "reason": "Catalog lookup skipped to save time"}
        match_status = episode_match_status(result, listing, int(known_season) if known_season else None)
        payload = {"result": result, "listing": listing, "match_status": match_status,
                   "evidence": evidence_label, "source_notes": added_notes, "url": clean_url or "",
                   "provider": used_provider, "user_season": int(known_season) if known_season else None,
                   "evidence_sources": evidence, "timestamp": time.time()}
        st.session_state["current_result"] = payload
        # Do not cache a Gemini fallback as an OpenAI success: it masks later fixes.
        st.session_state.setdefault("analysis_cache", {})
        if used_provider != "Gemini fallback":
            st.session_state["analysis_cache"][key] = payload
        # Cap session-memory usage and avoid persistent/cross-user caches.
        while len(st.session_state.analysis_cache) > 8:
            st.session_state.analysis_cache.pop(next(iter(st.session_state.analysis_cache)))
        st.session_state.pop("deep_result", None)

payload = st.session_state.get("current_result")
if payload:
    r = payload["result"]
    listing = payload["listing"]
    status = payload["match_status"]
    st.divider()
    st.subheader(safe_string(r.get("show"), 150) or "Show not identified")
    st.warning(f"**{status['label']}** — {status['reason']}")
    x, y, z = st.columns(3)
    x.metric("Proposed season", safe_string(r.get("season"), 12) or "Unknown")
    y.metric("Proposed episode", safe_string(r.get("episode"), 12) or "Unknown")
    z.metric("AI confidence (not proof)", safe_string(r.get("confidence"), 20).title() or "Unknown")
    st.write("**Proposed title (not confirmed from clip):** " + (safe_string(r.get("episode_title"), 250) or "Unknown"))
    st.caption(f"AI provider: {payload['provider']} · Input: {payload['evidence']}")
    st.write("**Scene evidence reason:** " + (safe_string(r.get("confidence_reason"), 1400) or "Not explained"))
    if listing.get("status") == "episode_listing_confirmed":
        st.info(f"Episode catalog confirms {listing['show']} S{listing['season']}E{listing['episode']} — {listing['title']} **exists**, NOT that this short comes from it.")
        if listing.get("url"):
            st.link_button("View episode listing", listing["url"])
    else:
        st.caption("TVMaze: " + safe_string(listing.get("reason"), 300))
    st.markdown("### Clip summary")
    st.write(safe_string(r.get("clip_summary"), 4200) or "Not enough scene evidence")
    if r.get("full_episode_context"):
        with st.expander("Possible episode context (hypothesis)"):
            st.write(safe_string(r.get("full_episode_context"), 2800))
    st.markdown("### Clues & creator notes")
    for label, field in (("Characters", "characters"), ("Dialogue clues", "dialogue_clues"),
                         ("Search queries", "search_queries"), ("Alternative matches", "alternative_matches")):
        items = r.get(field) or []
        if isinstance(items, list) and items:
            st.write(f"**{label}:** " + "; ".join(safe_string(v, 320) for v in items[:8]))
    st.write("**Editing notes:** " + safe_string(r.get("creator_notes"), 2200))
    st.write("**Caption idea:** " + safe_string(r.get("caption_idea"), 1200))
    tags = r.get("hashtags") or []
    if isinstance(tags, list):
        st.write("**Hashtags:** " + " ".join(safe_string(t, 50) for t in tags[:12]))
    st.caption("Limitations: " + safe_string(r.get("evidence_limitations"), 1200))

    st.divider()
    st.markdown("### Deep episode verification — optional")
    st.caption("Search distinct dialogue and scene details online; uses a separate potentially paid API request. An episode listing alone never verifies the clip.")
    if st.button("Search for matching scene evidence (extra API cost)", use_container_width=True):
        if st.session_state.get("deep_result"):
            st.info("Using previous deep research in this session.")
        else:
            try:
                if openai_key:
                    with st.spinner("Searching web for scene-specific episode evidence..."):
                        deep = research_openai(openai_key, openai_verify_model,
                                               r, payload["url"], payload["source_notes"], payload["user_season"])
                elif gemini_key:
                    with st.spinner("Searching web with Gemini..."):
                        deep = grounded_web_research(gemini_key, gemini_model,
                                                     r, payload["url"], payload["source_notes"])
                else:
                    raise RuntimeError("No provider for search")
                st.session_state["deep_result"] = deep
            except Exception as exc:
                st.error("Web verification unavailable: " + safe_string(exc, 350))
    if st.session_state.get("deep_result"):
        deep = st.session_state.deep_result
        st.markdown("#### Web research — candidate matches, not automatic proof")
        st.write(deep.get("text") or "No explanation returned.")
        if deep.get("sources"):
            for source in deep["sources"]:
                if source.get("url", "").startswith("https://"):
                    st.link_button(safe_string(source.get("title"), 90) or "Read source", source["url"])
        else:
            st.info("No citation URLs returned. Treat candidate as unverified.")

    st.link_button("Search exact episode on Google", "https://www.google.com/search?q=" + quote(
        f"{r.get('show') or ''} season {r.get('season') or ''} episode {r.get('episode') or ''} {r.get('episode_title') or ''}"))
    report = format_download(r, listing, payload["evidence"], payload["url"])
    st.download_button("Download research notes", report, "reachlab_episode_research.md", mime="text/markdown")

st.divider()
st.caption("ReachLab AI · Cloud Streamlit MVP · Scene IDs may be wrong; verify before clipping. Avoid sharing your API keys.")
