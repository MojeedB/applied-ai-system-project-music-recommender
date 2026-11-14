"""
Profiles saved by users from the app (a described request turned into a profile).

Stored in data/user_profiles.json as {title: profile}. Kept separate from profiles.py,
which is hand-written code. Plain Python (no Streamlit/Gemini) so it can be tested alone.
"""

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

USER_PROFILES_PATH = Path(__file__).resolve().parent.parent / "data" / "user_profiles.json"
MAX_TITLE_LEN = 40


def load_user_profiles(path: Optional[Path] = None) -> Dict[str, Dict]:
    """Saved profiles, or {} if the file is missing or unreadable."""
    path = path or USER_PROFILES_PATH
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(title): prof for title, prof in data.items() if isinstance(prof, dict)}


def validate_title(title: str, taken: Iterable[str]) -> Tuple[str, Optional[str]]:
    """Return (cleaned_title, error). `error` is None when the title is usable."""
    clean = " ".join(title.split())
    if not clean:
        return clean, "Please enter a title for the profile."
    if len(clean) > MAX_TITLE_LEN:
        return clean, f"Please keep the title to {MAX_TITLE_LEN} characters or fewer."
    if clean.lower() in {t.lower() for t in taken}:
        return clean, f"A profile called '{clean}' already exists. Please pick a different title."
    return clean, None


def save_user_profile(title: str, profile: Dict, reserved: Iterable[str] = (),
                      path: Optional[Path] = None) -> Optional[str]:
    """Save `profile` under `title`. Returns an error message, or None on success.

    `reserved` are titles that can't be used (the built-in profiles).
    """
    path = Path(path or USER_PROFILES_PATH)
    existing = load_user_profiles(path)
    clean, error = validate_title(title, list(reserved) + list(existing))
    if error:
        return error
    if not profile:
        return "That request didn't produce any preferences to save."

    existing[clean] = profile
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        os.replace(tmp, path)  # atomic, so a crash can't leave a half-written file
    except OSError:
        return "I couldn't save the profile to disk. Please try again."
    return None
