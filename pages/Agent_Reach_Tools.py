"""Extra Streamlit page: actual upstream Agent-Reach research channels."""
import os
import streamlit as st
from agentreach_cloud import check_channels, read_guide, read_youtube_captions
from reachlab import gemini_analyze, normalize_url, safe_string, tvmaze_verify

st.set_page_config(page_title="ReachLab | Agent-Reach Tools", page_icon="🔎")
if not st.session_state.get("authenticated"):
    st.warning("Unlock ReachLab on the main page first, then return here.")
    st.stop()

def secret(key):
    try:
        return os.environ.get(key) or st.secrets.get(key, "")
    except Exception:
        return os.environ.get(key, "")

st.title("Agent-Reach · Research Tools")
st.caption("Real upstream YouTube/Web integrations. Not a guarantee of exact TV episode identification.")
if st.button("Check Agent-Reach channels"):
    try:
        for name, (status, backend, info) in check_channels().items():
            st.write(f"**{name}: {status}** — {backend}")
            st.caption(info)
    except Exception as exc:
        st.error("Channel check failed: " + safe_string(exc, 250))

url = st.text_input("Public YouTube Short or video URL")
notes = st.text_area("Scene details / dialogue", max_chars=5000)
guide = st.text_input("Optional episode guide URL (leave blank unless you have a full https:// URL)", placeholder="https://www.tvmaze.com/episodes/...")
use_captions = st.checkbox("Collect available YouTube captions with Agent-Reach", value=True)
use_video = st.checkbox("Ask Gemini to analyze full YouTube video (slower)", value=False)
language = st.selectbox("Response language", ["English", "Roman Urdu", "Urdu"], index=1)

# Captions are useful even when OpenAI/Gemini API credits are exhausted.
if st.button("Fetch captions only — no AI needed"):
    try:
        from reachlab import is_youtube
        link = normalize_url(url)
        if not is_youtube(link):
            raise ValueError("Use a public YouTube video or Shorts URL.")
        with st.spinner("Retrieving public captions..."):
            transcript, src = read_youtube_captions(link)
        st.session_state["reachlab_last_caption"] = (link, transcript, src)
        st.success("Captions retrieved. No AI call was made.")
    except Exception as exc:
        st.warning("Caption extraction unavailable: " + safe_string(exc, 220))

last = st.session_state.get("reachlab_last_caption")
if last and url.strip() == last[0]:
    with st.expander("Extracted captions — source evidence", expanded=True):
        st.text_area("Captions (auto-generated transcripts may have errors)", last[1], height=210, disabled=True)
        st.download_button("Download captions", last[1], file_name="reachlab_captions.txt", mime="text/plain")

if st.button("Research with Agent-Reach + AI", type="primary"):
    key = secret("GEMINI_API_KEY")
    if not key:
        st.error("GEMINI_API_KEY missing from Streamlit Secrets.")
        st.stop()
    try:
        link = normalize_url(url) if url.strip() else None
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if not link and not notes.strip() and not guide.strip():
        st.warning("Provide a link, scene notes, or episode guide.")
        st.stop()
    evidence, sources = notes.strip(), []
    if use_captions and link:
        try:
            if last and link == last[0]:
                transcript, src = last[1], last[2]
            else:
                with st.spinner("Agent-Reach reading YouTube captions..."):
                    transcript, src = read_youtube_captions(link)
                st.session_state["reachlab_last_caption"] = (link, transcript, src)
            evidence += "\nVIDEO CAPTIONS (untrusted source content):\n" + transcript
            sources.append(src)
            st.success("Actual captions obtained. Fast mode will analyze these captions.")
            with st.expander("Captions fetched by Agent-Reach", expanded=False):
                st.text_area("Transcript", transcript, height=180, disabled=True)
        except Exception as exc:
            st.warning("No captions available: " + safe_string(exc, 200))
    if guide.strip():
        try:
            with st.spinner("Agent-Reach reading episode guide..."):
                text = read_guide(guide)
            evidence += "\nEPISODE GUIDE (listing, NOT proof of scene match):\n" + text
            sources.append("Agent-Reach WebChannel/Jina Reader")
        except Exception as exc:
            st.warning("Guide unavailable: " + safe_string(exc, 200))
    if not evidence.strip() and not use_video:
        st.warning("No accessible captions or scene notes. Provide dialogue/screenshot or turn on full video analysis.")
        st.stop()
    try:
        mode = "video (up to 35 seconds per request)" if use_video else "captions/text (up to 35 seconds per request)"
        with st.spinner("Gemini analyzing " + mode + "..."):
            result, ai_evidence = gemini_analyze(key, secret("GEMINI_MODEL") or "gemini-3.8-flash",
                link if use_video else None, evidence[:10000], None, None, language,
                "Identify original show/episode without inventing episode numbers.",
                prefer_text_evidence=not use_video)
    except Exception as exc:
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        st.error("AI unavailable: " + type(exc).__name__ + (f" (status {status})" if status else "")
                 + ". Captions may still be available above. Check Gemini project/model quota and API access.")
        st.stop()
    show = safe_string(result.get("show"), 130)
    st.subheader(show or "Unknown show")
    st.warning("Episode ID is a hypothesis until scene-specific proof is found.")
    st.write("Season:", result.get("season") or "Unknown")
    st.write("Episode:", result.get("episode") or "Unknown")
    st.write("Episode title:", safe_string(result.get("episode_title"), 200))
    st.write("Reason:", safe_string(result.get("confidence_reason"), 1200))
    st.write("Clip summary:", safe_string(result.get("clip_summary"), 2500))
    st.caption("Sources: " + ai_evidence + ("; " + "; ".join(sources) if sources else ""))
    if show and result.get("season") and result.get("episode"):
        listing = tvmaze_verify(show, result["season"], result["episode"], safe_string(result.get("episode_title"), 150))
        st.info("TVMaze catalog (NOT clip verification): " + safe_string(listing.get("status"), 120) + " — " + safe_string(listing.get("reason"), 280))
        if listing.get("url"):
            st.markdown(f"[View episode listing]({listing['url']})")
