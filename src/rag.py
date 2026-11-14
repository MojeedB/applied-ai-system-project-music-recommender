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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

import recommender
from profiles import PROFILES  # user-facing profiles only; ADVERSARIAL_PROFILES is never imported here
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


def _ask(system: str, user: str, max_tokens: int = 4096, json_mode: bool = False,
         temperature: Optional[float] = None, seed: Optional[int] = None) -> str:
    # Thinking models spend hidden reasoning tokens from this same budget, so keep it generous.
    # temperature/seed left as None means "use Gemini's defaults" (more varied wording).
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
        response_mime_type="application/json" if json_mode else None,
        temperature=temperature,
        seed=seed,
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


# --- Profile validation (shared by Gemini-built and pre-defined profiles) ---
def clean_profile(profile: Dict) -> Dict:
    """Keep only known keys with valid types and clamp numeric values into range.

    Used for both untrusted Gemini output and profiles from profiles.py, so a bad value
    (unknown key, wrong type, energy 1.8, tempo 400) can't distort scoring.
    """
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


# --- 1. Query understanding: free text -> the profile dict score_song expects ---
def text_to_profile(query: str, genres: List[str], moods: List[str]) -> Dict:
    system = (
        "Convert a music request into a JSON object using ONLY these optional keys:\n"
        f"{PROFILE_FIELDS.format(tempo_min=recommender.TEMPO_MIN, tempo_max=recommender.TEMPO_MAX)}\n"
        f"Prefer genres from: {sorted(genres)}\nPrefer moods from: {sorted(moods)}\n"
        "Omit any key the request gives no signal for. Output JSON only."
    )
    # Understanding step: be as repeatable as possible so the same request parses to the same
    # profile. (The answer-writing step keeps the default temperature for natural wording.)
    raw = _ask(system, query, json_mode=True, temperature=0, seed=0)
    match = re.search(r"\{.*\}", raw, re.DOTALL)  # tolerate stray prose/code fences
    try:
        profile = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        profile = {}

    return clean_profile(profile)


# --- 2. Retrieval: reuse the existing scorer ---
def retrieve(profile: Dict, songs: List[Dict], k: int = 5):
    return recommend_songs(profile, songs, k=k)


# --- 3. Augmented generation: only the retrieved songs go into the prompt ---
INF = float("inf")  # upper limit of the top band: "anything above the previous limit"


def _band(value: float, bands) -> str:
    """Return the first label whose upper limit is above `value` (bands: [(limit, label), ...]).

    Limits are ascending; the last band uses INF so it catches every remaining value.
    """
    for limit, label in bands:
        if value < limit:
            return label
    return bands[-1][1]


def describe_song(s: Dict) -> str:
    """Turn raw numeric features into everyday words so no backend numbers reach the user."""
    energy = _band(s["energy"], [(0.35, "very calm, low energy"), (0.55, "relaxed, gentle energy"),
                                 (0.75, "moderate energy"), (0.88, "high energy"),
                                 (INF, "very high energy")])
    tempo = _band(s["tempo_bpm"], [(75, "slow tempo"), (100, "mid tempo"), (125, "upbeat tempo"),
                                   (160, "fast tempo"), (INF, "very fast tempo")])
    acoustic = _band(s["acousticness"], [(0.2, "electronic, low acousticness"),
                                         (0.5, "a mix of electronic and acoustic sound"),
                                         (0.8, "mostly acoustic"), (INF, "very acoustic")])
    dance = _band(s["danceability"], [(0.4, "not very danceable"), (0.7, "moderately danceable"),
                                      (INF, "very danceable")])
    return f"{energy}; {tempo}; {acoustic}; {dance}"


def generate_answer(query: str, results) -> str:
    # Only words go into the prompt (no energy values, BPM or match scores), so the model
    # has no numbers to quote back at the user.
    context = "\n".join(
        f"[{i}] \"{s['title']}\" by {s['artist']} | {s['genre']}, {s['mood']} mood | "
        f"{describe_song(s)}"
        for i, (s, _score, _) in enumerate(results, start=1)
    )
    system = (
        "You are a friendly music recommender talking to a casual listener. Recommend songs "
        "ONLY from the numbered list provided; never mention any song that is not in it. "
        "For each, give one short sentence on why it fits the request, using everyday words "
        "like 'energetic', 'high tempo', 'mellow' or 'low acousticness'. Never write numbers, "
        "decimals, BPM values, scores or percentages. If the matches are weak, say so honestly."
    )
    return _ask(system, f"Request: {query}\n\nRetrieved songs:\n{context}")


# --- Pipeline ---
def _plain_list(results) -> str:
    """Non-AI fallback: the retrieved songs as a simple list."""
    lines = [f"  {i}. \"{s['title']}\" by {s['artist']} ({s['genre']}, {s['mood']})"
             for i, (s, _, _) in enumerate(results, start=1)]
    return "\n".join(lines)


# Parsed-profile key -> the song field it is matched against (in display order).
PROFILE_TO_SONG_FIELD = {
    "favorite_genre": "genre",
    "favorite_mood": "mood",
    "target_energy": "energy",
    "pref_tempo": "tempo_bpm",
    "target_danceable": "danceability",
    "target_acoustic": "acousticness",
}


def _backend_block(results, profile: Dict) -> str:
    """Backend/verification lines: for each chosen song (rank order), the values of exactly the
    fields the parsed profile asked for. If the profile is empty, every field is shown."""
    fields = [song_field for key, song_field in PROFILE_TO_SONG_FIELD.items() if key in profile]
    fields = fields or list(PROFILE_TO_SONG_FIELD.values())
    lines = []
    for rank, (s, _score, _) in enumerate(results, start=1):
        shown = ", ".join(f"{f!r}: {s[f]!r}" for f in fields)
        lines.append(f"  {rank}. \"{s['title']}\" by {s['artist']} {{{shown}}}")
    return "Backend data (chosen songs):\n" + "\n".join(lines)


@dataclass
class Recommendation:
    """Outcome of one recommendation request."""
    text: str                 # always a printable, user-friendly message
    ok: bool                  # False if a Gemini call failed (text is then an apology/fallback)
    profile: Optional[Dict] = None
    results: list = field(default_factory=list)


def normalize_query(query: str) -> str:
    """Lowercase and collapse whitespace so 'Chill  Music ' and 'chill music' are the same request."""
    return " ".join(query.lower().split())


def _recommend(profile: Dict, request: str, songs: List[Dict], k: int, label: str) -> Recommendation:
    """Shared retrieve -> generate -> fallback steps for a ready-made profile."""
    print(f"{label}: {profile}")
    results = retrieve(profile, songs, k=k)

    # Backend/verification output goes to the terminal only (like the profile line above),
    # never into the answer text the user sees in the app.
    print(_backend_block(results, profile))

    try:
        text = generate_answer(request, results)
        return Recommendation(text, True, profile, results)
    except GeminiError as err:
        # Retrieval worked, so still give the user useful results without the AI write-up.
        text = (f"I found some songs you might like, but I couldn't write up the "
                f"explanations this time. {err}\n\nHere are the best matches:\n"
                f"{_plain_list(results)}")
        return Recommendation(text, False, profile, results)


def get_recommendation_from_text(query: str, k: int = 5) -> Recommendation:
    """Never raises for Gemini problems; check `.ok` to tell a real answer from a fallback."""
    query = normalize_query(query)
    if not query:
        return Recommendation("Please describe what you'd like to hear first.", False)
    songs = load_songs(str(DATA_PATH))

    try:
        profile = text_to_profile(
            query,
            genres=list({s["genre"] for s in songs}),
            moods=list({s["mood"] for s in songs}),
        )
    except GeminiError as err:
        # Without the AI we can't turn free text into a taste profile, so we can't search.
        return Recommendation(f"Sorry, I couldn't work out your request this time. {err}", False)

    return _recommend(profile, query, songs, k, label="Parsed profile")


def recommend_from_text(query: str, k: int = 5) -> str:
    """Always returns a printable, user-friendly string, even when Gemini is unavailable."""
    return get_recommendation_from_text(query, k).text


def _profile_request(name: str, profile: Dict) -> str:
    """Describe a profile in words (no numbers) to stand in for the user's typed request."""
    parts = []
    if profile.get("favorite_genre"):
        parts.append(f"{profile['favorite_genre']} music")
    if profile.get("favorite_mood"):
        parts.append(f"a {profile['favorite_mood']} mood")
    if "target_energy" in profile:
        parts.append(_band(profile["target_energy"], [(0.35, "very calm energy"), (0.55, "gentle energy"),
                                                      (0.75, "moderate energy"), (0.88, "high energy"),
                                                      (INF, "very high energy")]))
    if "pref_tempo" in profile:
        parts.append(_band(profile["pref_tempo"], [(75, "a slow tempo"), (100, "a mid tempo"),
                                                   (125, "an upbeat tempo"), (160, "a fast tempo"),
                                                   (INF, "a very fast tempo")]))
    if "target_acoustic" in profile:
        parts.append(_band(profile["target_acoustic"], [(0.2, "an electronic sound"),
                                                        (0.5, "a mix of electronic and acoustic sound"),
                                                        (0.8, "a mostly acoustic sound"),
                                                        (INF, "a very acoustic sound")]))
    return f"A listener who picked the '{name}' style: " + ", ".join(parts)


def short_title(profile: Dict) -> str:
    """A short, number-free label for a profile, e.g. 'Lofi · Focused · Mellow'."""
    parts = []
    if profile.get("favorite_genre"):
        parts.append(profile["favorite_genre"].strip().title())
    if profile.get("favorite_mood"):
        parts.append(profile["favorite_mood"].strip().capitalize())
    if "target_energy" in profile:
        parts.append(_band(profile["target_energy"], [(0.35, "Calm"), (0.55, "Mellow"), (0.75, "Moderate"),
                                                      (0.88, "High energy"), (INF, "Max energy")]))
    return " · ".join(parts) or "Any music"


def get_recommendation_from_profile(name: str, k: int = 5,
                                    profiles: Optional[Dict[str, Dict]] = None) -> Recommendation:
    """Recommend from a named profile (skips the first Gemini call).

    `profiles` defaults to the user-facing PROFILES; pass a merged dict to include saved ones.
    """
    available = PROFILES if profiles is None else profiles
    if name not in available:
        return Recommendation(f"Sorry, I don't have a profile called '{name}'.", False)
    songs = load_songs(str(DATA_PATH))
    profile = clean_profile(available[name])
    return _recommend(profile, _profile_request(name, profile), songs, k, label="Profile")


def recommend_from_profile(name: str, k: int = 5) -> str:
    return get_recommendation_from_profile(name, k).text


def choose_and_recommend() -> str:
    """Menu: pick a ready-made profile or type your own request."""
    names = list(PROFILES)
    own = len(names) + 1
    print("Pick a profile or describe what you want:")
    for i, name in enumerate(names, start=1):
        print(f"  {i}. {name}")
    print(f"  {own}. Type my own request")

    while True:
        choice = input("> ").strip()
        if choice.isdigit() and 1 <= int(choice) <= own:
            break
        print(f"Please enter a number from 1 to {own}.")

    if int(choice) == own:
        return recommend_from_text(input("What do you feel like listening to? "))
    return recommend_from_profile(names[int(choice) - 1])


if __name__ == "__main__":
    if "GEMINI_API_KEY" not in os.environ:
        raise SystemExit(f"GEMINI_API_KEY not found. Add it to {PROJECT_ROOT / '.env'} "
                         "(free key: https://aistudio.google.com/apikey).")
    print(choose_and_recommend())
