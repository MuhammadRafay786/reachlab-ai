"""OpenAI Responses API integration for ReachLab. No credentials persisted to disk."""
from __future__ import annotations

import base64
from typing import Any

from reachlab import STRUCTURED_PROMPT, decode_json_response, safe_string

FAST_MODEL = "gpt-4.1-mini"
VERIFY_MODEL = "gpt-4.1"


def image_part(data: bytes, mime: str = "image/jpeg") -> dict:
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("Unsupported image MIME type")
    if len(data) > 7 * 1024 * 1024:
        raise ValueError("Image too large (7 MB maximum)")
    return {
        "type": "input_image",
        "image_url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}",
        "detail": "low" if len(data) > 700_000 else "auto",
    }


def analyze_openai(
    api_key: str, model: str, url: str | None, notes: str, language: str, goal: str,
    image: bytes | None = None, image_mime: str = "image/jpeg",
    video_frames: list[bytes] | None = None,
    known_season: int | None = None,
    client: Any = None,
) -> tuple[dict, str]:
    """Analyze ONLY supplied image/frames/text, never pretend a URL was watched.

    Caller may add public metadata/captions to notes, clearly labelled as such.
    """
    if not api_key and client is None:
        raise ValueError("OPENAI_API_KEY is missing")
    from openai import OpenAI

    client = client or OpenAI(api_key=api_key, timeout=45.0, max_retries=1)
    frames = (video_frames or [])[:8]
    has_visual = bool(frames or image)
    text = (
        f"LINK (reference only, not actual video frames): {url or 'none'}\n"
        f"KNOWN SEASON USER HINT (not verified): {known_season or 'none'}\n"
        f"EVIDENCE / USER NOTES / PUBLIC CAPTIONS:\n{notes[:12000]}\n"
        f"VISUAL CONTENT ATTACHED: {len(frames)} sampled frames; one screenshot: {bool(image)}\n"
        "Do not claim to have viewed a YouTube video based solely on a URL. "
        "Visuals are samples, not the entire video. Captions may be imperfect. "
        "If only metadata is supplied with no scene clues, leave season/episode null. "
        "Return valid JSON matching all fields in the instructions."
    )
    content = [{"type": "input_text", "text": text}]
    if image:
        content.append(image_part(image, image_mime))
    for f in frames:
        content.append(image_part(f))
    instructions = STRUCTURED_PROMPT.format(language=language, goal=goal[:300])
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=[{"role": "user", "content": content}],
        max_output_tokens=1400,
        store=False,
    )
    answer = (getattr(response, "output_text", None) or "").strip()
    if not answer:
        raise RuntimeError("OpenAI returned no text. Check account credits and model access.")
    parsed = decode_json_response(answer)
    label = (f"{len(frames)} sampled video frames" if frames else
             "Scene screenshot" if image else
             "Available text/captions/metadata only — no video frames")
    if image and frames:
        label += " + screenshot"
    if not has_visual and not notes.strip():
        label = "URL only (episode cannot be verified)"
    return parsed, label + f" · {model}"


def research_openai(api_key: str, model: str, prediction: dict,
                    url: str, notes: str, known_season: int | None = None,
                    client: Any = None) -> dict:
    """An opt-in, separate billed web search. Search hits do not prove origin."""
    if not api_key and client is None:
        raise ValueError("OPENAI_API_KEY is missing")
    from openai import OpenAI
    client = client or OpenAI(api_key=api_key, timeout=65.0, max_retries=1)
    show = safe_string(prediction.get("show"), 100)
    clue_list = prediction.get("dialogue_clues") or []
    if not isinstance(clue_list, list):
        clue_list = []
    prompt = (
        "Investigate which exact SHOW EPISODE a short clip likely came from using web search. "
        "Search distinctive dialogue quotes, scene-specific synopses, and reputable episode transcripts FIRST. "
        "Treat the initial AI prediction as a hypothesis, not the answer. "
        "TVMaze or Wikipedia merely listing an episode is NOT evidence of this clip's origin. "
        "Do not invent quotes, website URLs, verified status, or timestamps. "
        "If scene-level published evidence cannot be found, state 'UNVERIFIED'. "
        "Return: best hypotheses (max 3), scene-level evidence FOR or AGAINST each, "
        "URLs cited inline, mismatches/limitations, and further search ideas. "
        f"Show hypothesis: {show or 'unknown'}\n"
        f"Known-season user hint: {known_season or 'none'}\n"
        f"Season/episode/title guess: {prediction.get('season')}/"
        f"{prediction.get('episode')}/{safe_string(prediction.get('episode_title'), 150)}\n"
        f"Scene summary: {safe_string(prediction.get('clip_summary'), 1200)}\n"
        f"Dialogue clues: {'; '.join(safe_string(x, 140) for x in clue_list[:8])}\n"
        f"Supplied notes/captions: {notes[:2800]}\n"
        f"Clip link (not evidence of scene identity): {url[:300]}"
    )
    response = client.responses.create(
        model=model,
        tools=[{"type": "web_search", "search_context_size": "low"}],
        input=prompt,
        max_output_tokens=1200,
        store=False,
    )
    answer = (getattr(response, "output_text", None) or "").strip()
    if not answer:
        raise RuntimeError("Search produced no response.")
    sources = []
    for item in getattr(response, "output", []) or []:
        for block in getattr(item, "content", []) or []:
            for citation in getattr(block, "annotations", []) or []:
                link = getattr(citation, "url", "") or ""
                if link.startswith("https://") and link not in [x["url"] for x in sources]:
                    sources.append({"title": safe_string(getattr(citation, "title", "Source"), 180), "url": link})
    return {"text": answer, "sources": sources[:12]}
