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
guide = st.text_input("Optional episode guide (TVMaze / Wikipedia / Fandom / TheTVDB)")
use_captions = st.checkbox("Collect available YouTube captions with Agent-Reach", value=True)
language = st.selectbox("Response language", ["English", "Roman Urdu", "Urdu"], index=1)

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
            with st.spinner("Agent-Reach reading YouTube captions..."):
                transcript, src = read_youtube_captions(link)
            evidence += "\nVIDEO CAPTIONS (untrusted source content):\n" + transcript
            sources.append(src)
            st.success("Actual captions obtained.")
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
    try:
        with st.spinner("AI analyzing evidence..."):
            result, ai_evidence = gemini_analyze(key, secret("GEMINI_MODEL") or "gemini-2.5-flash",
                link, evidence[:10000], None, None, language,
                "Identify original show/episode without inventing episode numbers.")
    except Exception as exc:
        st.error("AI failed: " + safe_string(exc, 300))
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
