"""ReachLab AI — Streamlit Community Cloud friendly episode finder MVP."""
import hmac
import os
import time
from urllib.parse import quote

import streamlit as st
from reachlab import (format_download, gemini_analyze, normalize_url,
                      public_media_metadata, safe_string, tvmaze_verify)

st.set_page_config(page_title="ReachLab AI | Find the Episode", page_icon="🎬", layout="centered")
st.markdown("""<style>
.block-container {max-width: 920px; padding-top: 1.75rem; padding-bottom:3rem;}
[data-testid="stApp"]{background: radial-gradient(ellipse at top left,rgba(86,90,225,.09),transparent 48%);}
.hero{padding: 24px 0 18px;}
.kicker{font-weight:800;letter-spacing:.16em;font-size:.72rem;color:#9695ff;}
.hero h1{font-size:3.05rem;line-height:1.13;letter-spacing:-.055em;margin:12px 0 10px;}
.hero p{font-size:1.03rem;color:#9397a7;max-width:760px;}
.info{padding:16px 18px;border-radius:12px;background:rgba(120,125,140,.09);
    border:1px solid rgba(130,135,150,.2);margin:12px 0;}
.small{color:#9ba0b2;font-size:.85rem;}
</style>""", unsafe_allow_html=True)


def get_secret(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    try:
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""


pwd = get_secret("APP_PASSWORD")
api_key = get_secret("GEMINI_API_KEY")
model = get_secret("GEMINI_MODEL") or "gemini-3.8-flash"

if not pwd or len(pwd) < 12:
    st.error("APP_PASSWORD missing or too short. Set a unique 12+ character password in Streamlit Cloud > App settings > Secrets.")
    st.stop()

if not st.session_state.get("authenticated"):
    st.markdown('<div class="hero"><div class="kicker">PRIVATE RESEARCH WORKSPACE</div><h1>ReachLab<span style="color:#9a98ff">.</span></h1><p>Find the original episode behind a social clip.</p></div>', unsafe_allow_html=True)
    st.text_input("Access password", type="password", key="input_password")
    if st.button("Unlock", type="primary", use_container_width=True):
        fails = st.session_state.get("fails", 0)
        if fails >= 8:
            st.error("Too many attempts in this session. Restart your browser session later.")
        elif hmac.compare_digest(st.session_state.input_password, pwd):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.session_state.fails = fails + 1
            st.error("Incorrect password.")
    st.stop()

st.markdown('<div class="hero"><div class="kicker">CLOUD EPISODE RESEARCH ASSISTANT</div><h1>Find the scene. <span style="color:#a8a6ff">Find the episode.</span></h1><p>Paste a YouTube Short, Reel or Reddit link. Identify possible show, season, episode and gather editing notes—without pretending guesses are verified facts.</p></div>', unsafe_allow_html=True)

if not api_key:
    st.warning("Cloud AI key is not configured. Add GEMINI_API_KEY to Streamlit Secrets; do not put it in GitHub.")

with st.sidebar:
    st.header("Research settings")
    language = st.selectbox("Output language", ["English", "Roman Urdu", "Urdu"], index=1)
    goal = st.selectbox("Goal", ["Identify the original TV/cartoon episode", "Summarize scene for a clipping channel", "Generate caption, hashtags and research notes"])
    st.divider()
    st.caption("Cloud provider: Gemini. Uses free quota where available; API usage and regional limits depend on your Google project.")
    st.caption("Agent-Reach is a CLI toolkit, not an internet-facing API. This cloud MVP uses direct official video AI and optional public metadata extraction.")
    st.caption("Copyright: Find the authorized source episode; check licensing before publishing third-party clips.")

url = st.text_input("Paste a public social link", placeholder="https://www.youtube.com/shorts/...")
scene_notes = st.text_area("Optional scene description or dialogue", placeholder="Example: Peter and Stewie are talking about ...", max_chars=10000, height=110)
shot = st.file_uploader("Optional screenshot of the scene (PNG/JPG, max 7 MB)", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)
with col1:
    analyze = st.button("Analyze & find episode", type="primary", use_container_width=True, disabled=not bool(api_key))
with col2:
    st.caption("YouTube: actual video analysis where API access allows. Other sites: notes, image and best-effort public metadata.")

if analyze:
    if time.monotonic() - st.session_state.get("last_run", -1e10) < 15:
        st.warning("Please wait at least 15 seconds between requests.")
        st.stop()
    st.session_state.last_run = time.monotonic()
    try:
        link = normalize_url(url) if url.strip() else None
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if not link and not scene_notes.strip() and not shot:
        st.error("Please provide a link, scene description or screenshot.")
        st.stop()
    shot_bytes, mime = None, None
    if shot:
        if shot.size > 7 * 1024 * 1024:
            st.error("Screenshot is too large. Max 7 MB.")
            st.stop()
        shot_bytes = shot.getvalue()
        mime = "image/png" if shot.type == "image/png" else "image/jpeg"

    additional = scene_notes.strip()
    if link and not ("youtube.com" in link or "youtu.be" in link):
        try:
            with st.spinner("Reading public post metadata (best effort)..."):
                meta = public_media_metadata(link)
            additional += ("\nPUBLIC METADATA (not video evidence):\n"
                           + f"Title: {meta['title']}\nCreator: {meta['creator']}\nDescription: {meta['description']}")
            st.info("Public post metadata found. The app has not watched Instagram/Reddit video; episode ID will be a hypothesis.")
        except Exception:
            st.info("Instagram/Reddit restricts this content from the cloud. Please add a screenshot or describe the scene.")
    try:
        with st.spinner("Analyzing evidence with Gemini..."):
            result, evidence = gemini_analyze(api_key, model, link, additional, shot_bytes, mime, language, goal)
    except Exception as exc:
        st.error("Analysis failed: " + safe_string(exc, 500))
        st.stop()

    show = safe_string(result.get("show"), 130)
    season, episode = result.get("season"), result.get("episode")
    proposed_title = safe_string(result.get("episode_title"), 220)
    with st.spinner("Cross-checking episode listing in TVMaze..."):
        checked = tvmaze_verify(show, season, episode, proposed_title)

    st.success("Research completed. Verify the clip source before publishing.")
    st.subheader(show or "Show not confidently identified")
    a, b, c = st.columns(3)
    a.metric("Season", str(season or "Unknown"))
    b.metric("Episode", str(episode or "Unknown"))
    c.metric("AI confidence", safe_string(result.get("confidence"), 20).title() or "Unknown")
    st.write("**Proposed episode title:** " + (proposed_title or "Unknown"))
    st.write("**Evidence type:** " + evidence)
    st.write("**Why this match:** " + safe_string(result.get("confidence_reason"), 1600))

    if checked["status"] == "episode_listing_confirmed":
        st.info(f"Episode catalog confirmed: **{checked['show']} — S{checked['season']}E{checked['episode']} — {checked['title']}**. This does **not** verify the clip originated from this episode.")
        if checked.get("url"):
            st.markdown("[View episode listing in TVMaze]({})".format(checked["url"]))
        if checked.get("summary"):
            with st.expander("TVMaze listed episode summary"):
                st.write(checked["summary"])
    else:
        st.warning("Not independently verified: " + checked.get("reason", ""))

    st.markdown("### Clip summary")
    st.write(safe_string(result.get("clip_summary"), 4000) or "Insufficient evidence.")
    if result.get("full_episode_context"):
        st.markdown("### Possible episode context")
        st.write(safe_string(result["full_episode_context"], 3000))
    st.markdown("### Research & editing")
    for heading, field in (("Characters", "characters"), ("Dialogue clues", "dialogue_clues"),
                           ("Suggested searches", "search_queries"), ("Alternate matches", "alternative_matches")):
        xs = result.get(field) or []
        if isinstance(xs, list) and xs:
            st.write(f"**{heading}:** " + "; ".join(safe_string(x, 300) for x in xs[:8]))
    st.write("**Editing notes:** " + safe_string(result.get("creator_notes"), 2500))
    st.write("**Caption idea:** " + safe_string(result.get("caption_idea"), 1200))
    tags = result.get("hashtags") or []
    if isinstance(tags, list):
        st.write("**Hashtags:** " + " ".join(safe_string(v, 40) for v in tags[:12]))
    st.caption("Limitations: " + safe_string(result.get("evidence_limitations"), 1400))
    if show:
        query = quote(f"{show} season {season or ''} episode {episode or ''} {proposed_title or ''}".strip())
        st.link_button("Search episode on Google", "https://www.google.com/search?q=" + query)
    report = format_download(result, checked, evidence, link or "")
    st.download_button("Download research notes", report, "reachlab_episode_research.md", mime="text/markdown")

st.divider()
st.caption("ReachLab MVP · Hosted with Streamlit Community Cloud · Private app password · Source identification is best-effort")
