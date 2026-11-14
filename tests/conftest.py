"""
Shared test setup.

- Puts src/ on sys.path so tests can import rag, profiles, recommender, ... the same way the
  app does (the original test_recommender.py imports `src.recommender`, so the project root
  stays importable too).
- If google-genai / python-dotenv aren't installed, stands in simple stubs so rag.py can be
  imported. No test ever calls the real Gemini API: tests replace `rag._ask`.
"""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import dotenv  # noqa: F401
except ImportError:
    stub = types.ModuleType("dotenv")
    stub.load_dotenv = lambda *a, **k: False
    sys.modules["dotenv"] = stub

try:
    from google import genai  # noqa: F401
    from google.genai import types as _genai_types  # noqa: F401
except ImportError:
    genai_stub = types.ModuleType("google.genai")
    types_stub = types.ModuleType("google.genai.types")
    types_stub.GenerateContentConfig = lambda **kwargs: kwargs
    genai_stub.Client = object
    genai_stub.types = types_stub
    google_mod = sys.modules.get("google") or types.ModuleType("google")
    google_mod.genai = genai_stub
    sys.modules.update({"google": google_mod, "google.genai": genai_stub,
                        "google.genai.types": types_stub})

import recommender  # noqa: E402  (after sys.path setup)

DATA_PATH = ROOT / "data" / "songs.csv"


@pytest.fixture(scope="session")
def catalog():
    """The real song catalog (data/songs.csv)."""
    return recommender.load_songs(str(DATA_PATH))


@pytest.fixture(autouse=True)
def _restore_tempo_range():
    """load_songs/set_tempo_range change module globals; put them back after every test."""
    saved = (recommender.TEMPO_MIN, recommender.TEMPO_MAX)
    yield
    recommender.TEMPO_MIN, recommender.TEMPO_MAX = saved
