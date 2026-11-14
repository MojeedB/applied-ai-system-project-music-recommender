"""
Minimal RAG layer for the music recommender (Gemini free tier).

  text query -> [LLM: query understanding] -> profile dict
             -> [retrieval: recommend_songs] -> top-k songs
             -> [LLM: grounded generation]   -> natural-language answer

Requires: pip install google-genai
          GEMINI_API_KEY in the environment (free key: https://aistudio.google.com/apikey)
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai import types

import recommender
from recommender import load_songs, recommend_songs

MODEL = "gemini-3.6-flash"  # confirm the current free-tier model name in AI Studio
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "songs.csv"

# Load GEMINI_API_KEY from applied-ai-system-final/.env (works from any working directory).
load_dotenv(PROJECT_ROOT / ".env")

PROFILE_FIELDS = """\
favorite_genre (string), favorite_mood (string),
target_energy (0-1), pref_tempo (bpm, {tempo_min}-{tempo_max}),
target_danceable (0-1), target_acoustic (0-1)"""

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()  # reads GEMINI_API_KEY
    return _client


class GeminiError(Exception):
    """A Gemini call failed. str(exc) is a message that is safe to show to the user."""


def _friendly_message(exc: Exception) -> str:
    """Translate an API/network failure into a short, non-technical explanation."""
    code = getattr(exc, "code", None)
    text = str(exc).lower()
    if code == 429 or "quota" in text or "resource_exhausted" in text:
        return ("The AI service is getting a lot of requests right now (free-tier limit reached). "
                "Please wait a minute and try again.")
    if code in (401, 403) or "api key" in text or "permission" in text:
        return ("I couldn't sign in to the AI service.")
        # Please check that GEMINI_API_KEY in your "
                # ".env file is correct and still active.")
    if code == 404 or "not found" in text:
        return ("The AI model I'm configured to use isn't available.")
                # "Please check the model name (MODEL) in rag.py.")
    if code == 400:
        return "The AI service couldn't understand that request. Try rephrasing what you want to hear."
    if isinstance(code, int) and code >= 500:
        return "The AI service is temporarily unavailable on Google's side. Please try again shortly."
    if isinstance(exc, (ConnectionError, TimeoutError)) or any(
            w in text for w in ("connect", "timeout", "timed out", "network", "dns", "ssl")):
        return "I couldn't reach the AI service. Please check your internet connection and try again."
    return "Something went wrong while contacting the AI service. Please try again in a moment."


def _ask(system: str, user: str, max_tokens: int = 4096, json_mode: bool = False) -> str:
    # Thinking models spend hidden reasoning tokens from this same budget, so keep it generous.
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        response_mime_type="application/json" if json_mode else None,
    )
    try:
        resp = _get_client().models.generate_content(model=MODEL, contents=user, config=config)
    except Exception as exc:  # SDK, HTTP and network errors all end up here
        logger.debug("Gemini call failed", exc_info=True)  # set logging to DEBUG to see details
        raise GeminiError(_friendly_message(exc)) from exc

    if resp.candidates:
        reason = str(resp.candidates[0].finish_reason)
        if "MAX_TOKENS" in reason:
            logger.warning("Response truncated (finish_reason=%s); raise max_tokens.", reason)
        elif "SAFETY" in reason or "PROHIBITED" in reason:
            raise GeminiError("The AI service declined to answer that request. "
                              "Try describing the music you want in different words.")
    text = resp.text
    if not text:
        raise GeminiError("The AI service sent back an empty reply. Please try again.")
    return text


# --- 1. Query understanding: free text -> the profile dict score_song expects ---
def text_to_profile(query: str, genres: List[str], moods: List[str]) -> Dict:
    system = (
        "Convert a music request into a JSON object using ONLY these optional keys:\n"
        f"{PROFILE_FIELDS.format(tempo_min=recommender.TEMPO_MIN, tempo_max=recommender.TEMPO_MAX)}\n"
        f"Prefer genres from: {sorted(genres)}\nPrefer moods from: {sorted(moods)}\n"
        "Omit any key the request gives no signal for. Output JSON only."
    )
    raw = _ask(system, query, json_mode=True)
    match = re.search(r"\{.*\}", raw, re.DOTALL)  # tolerate stray prose/code fences
    try:
        profile = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        profile = {}

    # Validate: keep known keys, clamp numeric ranges.
    clean: Dict = {}
    for key in ("favorite_genre", "favorite_mood"):
        if isinstance(profile.get(key), str):
            clean[key] = profile[key]
    for key, lo, hi in (("target_energy", 0, 1), ("target_danceable", 0, 1),
                        ("target_acoustic", 0, 1), ("pref_tempo", recommender.TEMPO_MIN, recommender.TEMPO_MAX)):
        value = profile.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            clean[key] = min(max(value, lo), hi)
    return clean


# --- 2. Retrieval: reuse the existing scorer ---
def retrieve(profile: Dict, songs: List[Dict], k: int = 5):
    return recommend_songs(profile, songs, k=k)


# --- 3. Augmented generation: only the retrieved songs go into the prompt ---
def generate_answer(query: str, results) -> str:
    context = "\n".join(
        f"[{i}] \"{s['title']}\" by {s['artist']} | genre={s['genre']}, mood={s['mood']}, "
        f"energy={s['energy']}, tempo={s['tempo_bpm']}bpm, acousticness={s['acousticness']} "
        f"| match score {score:.2f}"
        for i, (s, score, _) in enumerate(results, start=1)
    )
    system = (
        "You are a friendly music recommender. Recommend songs ONLY from the "
        "numbered list provided; never mention any song that is not in it. "
        "For each, give one sentence on why it fits the request, citing its "
        "attributes. If the matches are weak, say so honestly."
    )
    return _ask(system, f"Request: {query}\n\nRetrieved songs:\n{context}")


# --- Pipeline ---
def _plain_list(results) -> str:
    """Non-AI fallback: the retrieved songs as a simple list."""
    lines = [f"  {i}. \"{s['title']}\" by {s['artist']} ({s['genre']}, {s['mood']})"
             for i, (s, _, _) in enumerate(results, start=1)]
    return "\n".join(lines)


def recommend_from_text(query: str, k: int = 5) -> str:
    """Always returns a printable, user-friendly string, even when Gemini is unavailable."""
    songs = load_songs(str(DATA_PATH))

    try:
        profile = text_to_profile(
            query,
            genres=list({s["genre"] for s in songs}),
            moods=list({s["mood"] for s in songs}),
        )
    except GeminiError as err:
        # Without the AI we can't turn free text into a taste profile, so we can't search.
        return f"Sorry, I couldn't work out your request this time. {err}"

    results = retrieve(profile, songs, k=k)

    try:
        return generate_answer(query, results)
    except GeminiError as err:
        # Retrieval worked, so still give the user useful results without the AI write-up.
        return (f"I found some songs you might like, but I couldn't write up the "
                f"explanations this time. {err}\n\nHere are the best matches:\n"
                f"{_plain_list(results)}")


if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit(f"GEMINI_API_KEY not found. Add it to {PROJECT_ROOT / '.env'} "
                         "(free key: https://aistudio.google.com/apikey).")
    print(recommend_from_text(input("What do you feel like listening to? ")))
