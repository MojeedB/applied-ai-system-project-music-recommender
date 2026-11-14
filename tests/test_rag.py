"""rag.py: validation, wording helpers, error messages and the recommendation flows.

No test calls Gemini. `rag._ask` is replaced by a fake that records every call.
"""

import re

import pytest

import profiles as P
import rag

DIGIT = re.compile(r"\d")


class FakeAsk:
    """Stand-in for rag._ask. Returns canned text and records each call."""

    def __init__(self):
        self.calls = []
        self.profile_json = '{"favorite_genre": "lofi", "favorite_mood": "focused", "target_energy": 0.3}'
        self.answer = "ANSWER"
        self.fail = set()                 # {"understand"} and/or {"generate"}

    def __call__(self, system, user, **kwargs):
        step = "understand" if kwargs.get("json_mode") else "generate"
        self.calls.append({"step": step, "system": system, "user": user, **kwargs})
        if step in self.fail:
            raise rag.GeminiError("The AI service is getting a lot of requests.")
        return self.profile_json if step == "understand" else self.answer

    def steps(self):
        return [c["step"] for c in self.calls]


@pytest.fixture
def fake_ask(monkeypatch):
    fake = FakeAsk()
    monkeypatch.setattr(rag, "_ask", fake)
    return fake


# ---------- clean_profile ----------

def test_clean_profile_clamps_numbers_and_drops_junk():
    cleaned = rag.clean_profile({
        "favorite_genre": "pop", "target_energy": 1.8, "pref_tempo": 400,
        "target_danceable": -0.5, "target_acoustic": True,    # bool is not a number
        "bogus": 1, "favorite_mood": 5,                        # unknown key / wrong type
    })
    import recommender
    assert cleaned == {"favorite_genre": "pop", "target_energy": 1, "target_danceable": 0,
                       "pref_tempo": recommender.TEMPO_MAX}


def test_clean_profile_keeps_valid_values_untouched():
    profile = {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.4,
               "pref_tempo": 80, "target_danceable": 0.6, "target_acoustic": 0.8}
    assert rag.clean_profile(profile) == profile


def test_clean_profile_handles_empty_input():
    assert rag.clean_profile({}) == {}


def test_clean_profile_tempo_follows_the_dynamic_range():
    import recommender
    recommender.set_tempo_range([{"tempo_bpm": 100}, {"tempo_bpm": 200}])
    assert rag.clean_profile({"pref_tempo": 50})["pref_tempo"] == 100
    assert rag.clean_profile({"pref_tempo": 500})["pref_tempo"] == 200


def test_every_user_facing_profile_survives_cleaning_unchanged():
    for name, profile in P.PROFILES.items():
        assert rag.clean_profile(profile) == profile, name


# ---------- normalize_query ----------

@pytest.mark.parametrize("raw, expected", [
    ("Chill  MUSIC ", "chill music"), ("  upbeat\tsongs\n", "upbeat songs"), ("   ", ""), ("", ""),
])
def test_normalize_query(raw, expected):
    assert rag.normalize_query(raw) == expected


# ---------- _band / describe_song ----------

def test_band_limits_are_exclusive_upper_bounds():
    bands = [(0.35, "low"), (0.55, "mid"), (rag.INF, "high")]
    assert rag._band(0.0, bands) == "low"
    assert rag._band(0.34, bands) == "low"
    assert rag._band(0.35, bands) == "mid"          # the limit itself belongs to the next band
    assert rag._band(0.55, bands) == "high"
    assert rag._band(1.0, bands) == "high"
    assert rag._band(10 ** 9, bands) == "high"


def test_describe_song_never_contains_a_number_for_any_catalog_song(catalog):
    for song in catalog:
        assert not DIGIT.search(rag.describe_song(song)), song["title"]


def test_describe_song_covers_extremes_with_everyday_words():
    low = {"energy": 0.0, "tempo_bpm": 0, "acousticness": 0.0, "danceability": 0.0}
    high = {"energy": 1.0, "tempo_bpm": 300, "acousticness": 1.0, "danceability": 1.0}
    assert rag.describe_song(low) == ("very calm, low energy; slow tempo; "
                                      "electronic, low acousticness; not very danceable")
    assert rag.describe_song(high) == ("very high energy; very fast tempo; "
                                       "very acoustic; very danceable")


def test_describe_song_separates_fast_from_very_fast_tempo():
    base = {"energy": 0.5, "acousticness": 0.5, "danceability": 0.5}
    assert "upbeat tempo" in rag.describe_song({**base, "tempo_bpm": 120})
    assert "fast tempo" in rag.describe_song({**base, "tempo_bpm": 140})
    assert "very fast tempo" in rag.describe_song({**base, "tempo_bpm": 172})


# ---------- short_title / _profile_request ----------

def test_short_title_uses_parsed_profile_fields():
    assert rag.short_title({"favorite_genre": "lofi", "favorite_mood": "focused",
                            "target_energy": 0.3}) == "Lofi · Focused · Calm"
    assert rag.short_title({"favorite_genre": "r&b"}) == "R&B"
    assert rag.short_title({"target_energy": 0.95}) == "Max energy"
    assert rag.short_title({}) == "Any music"


def test_short_title_never_contains_digits():
    for profile in P.PROFILES.values():
        assert not DIGIT.search(rag.short_title(profile))


def test_profile_request_describes_the_profile_in_words_only():
    for name, profile in P.PROFILES.items():
        text = rag._profile_request(name, profile)
        assert text.startswith(f"A listener who picked the '{name}' style")
        assert not DIGIT.search(text.split(": ", 1)[1]), text


# ---------- _backend_block ----------

SONG = {"title": "Iron Requiem", "artist": "Black Harbor", "genre": "metal", "mood": "angry",
        "energy": 0.98, "tempo_bpm": 168, "danceability": 0.45, "acousticness": 0.04}


def test_backend_block_shows_only_fields_in_the_parsed_profile_with_title_and_artist():
    block = rag._backend_block([(SONG, 0.9, "")], {"favorite_genre": "metal", "target_energy": 0.9})
    assert block.splitlines()[0] == "Backend data (chosen songs):"
    assert block.splitlines()[1] == "  1. \"Iron Requiem\" by Black Harbor {'genre': 'metal', 'energy': 0.98}"
    assert "mood" not in block and "tempo" not in block


def test_backend_block_field_order_is_fixed_not_profile_order():
    profile = {"target_acoustic": 0.1, "favorite_mood": "angry", "pref_tempo": 170}
    line = rag._backend_block([(SONG, 0.9, "")], profile).splitlines()[1]
    assert line.index("'mood'") < line.index("'tempo_bpm'") < line.index("'acousticness'")


def test_backend_block_shows_every_field_for_an_empty_profile():
    line = rag._backend_block([(SONG, 0.9, "")], {}).splitlines()[1]
    for field in ("genre", "mood", "energy", "tempo_bpm", "danceability", "acousticness"):
        assert f"'{field}'" in line


def test_backend_block_numbers_songs_in_rank_order():
    other = {**SONG, "title": "Second Song"}
    lines = rag._backend_block([(SONG, 0.9, ""), (other, 0.8, "")], {"favorite_genre": "metal"}).splitlines()
    assert lines[1].startswith("  1.") and "Iron Requiem" in lines[1]
    assert lines[2].startswith("  2.") and "Second Song" in lines[2]


# ---------- friendly error messages ----------

class ApiError(Exception):
    def __init__(self, code, message=""):
        super().__init__(message)
        self.code = code


@pytest.mark.parametrize("exc, fragment", [
    (ApiError(429), "lot of requests"),
    (ApiError(None, "RESOURCE_EXHAUSTED quota"), "lot of requests"),
    (ApiError(403), "sign in"),
    (ApiError(401), "sign in"),
    (ApiError(404), "isn't available"),
    (ApiError(400), "rephras"),
    (ApiError(503), "temporarily unavailable"),
    (ConnectionError("connect failed"), "internet connection"),
    (TimeoutError("timed out"), "internet connection"),
    (RuntimeError("???"), "Something went wrong"),
])
def test_friendly_message_is_plain_english(exc, fragment):
    message = rag._friendly_message(exc)
    assert fragment in message
    assert not DIGIT.search(message)          # never leak status codes to the user


# ---------- text_to_profile ----------

def test_text_to_profile_is_called_with_temperature_zero_and_json_mode(fake_ask):
    rag.text_to_profile("chill music", genres=["lofi"], moods=["chill"])
    call = fake_ask.calls[0]
    assert call["json_mode"] is True
    assert call["temperature"] == 0 and call["seed"] == 0


def test_text_to_profile_prompt_lists_catalog_vocabulary_and_live_tempo_range(fake_ask):
    import recommender
    recommender.set_tempo_range([{"tempo_bpm": 61}, {"tempo_bpm": 199}])
    rag.text_to_profile("x", genres=["lofi", "jazz"], moods=["chill"])
    system = fake_ask.calls[0]["system"]
    assert "'jazz'" in system and "'chill'" in system
    assert "61-199" in system


def test_text_to_profile_clamps_out_of_range_values(fake_ask):
    fake_ask.profile_json = '{"favorite_genre": "pop", "target_energy": 7, "target_danceable": -2}'
    assert rag.text_to_profile("x", [], []) == {"favorite_genre": "pop", "target_energy": 1,
                                                "target_danceable": 0}


def test_text_to_profile_tolerates_prose_around_json_and_bad_json(fake_ask):
    fake_ask.profile_json = 'Sure! ```json\n{"favorite_mood": "chill"}\n``` hope that helps'
    assert rag.text_to_profile("x", [], []) == {"favorite_mood": "chill"}
    fake_ask.profile_json = "{not valid json"
    assert rag.text_to_profile("x", [], []) == {}
    fake_ask.profile_json = "no json at all"
    assert rag.text_to_profile("x", [], []) == {}


# ---------- generate_answer ----------

def test_generate_answer_prompt_has_words_only_and_default_temperature(fake_ask):
    songs = [({"title": "Calm Waters", "artist": "Quiet Band", "genre": "ambient", "mood": "peaceful",
               "energy": 0.21, "tempo_bpm": 58, "acousticness": 0.97, "danceability": 0.3}, 0.93, "")]
    rag.generate_answer("something calm", songs)
    call = fake_ask.calls[0]
    assert "Calm Waters" in call["user"] and "very calm" in call["user"]
    song_text = re.sub(r"^\[\d+\]", "", call["user"].split("Retrieved songs:")[1].strip(), flags=re.M)
    assert not DIGIT.search(song_text)                                   # no energy/BPM/score numbers
    assert "temperature" not in call and "seed" not in call              # answer keeps default wording
    assert "Never write numbers" in call["system"]


# ---------- full flows (Gemini faked) ----------

def test_text_flow_success_prints_terminal_output_but_returns_clean_answer(fake_ask, capsys):
    rec = rag.get_recommendation_from_text("  Focus   MUSIC ", k=3)
    out = capsys.readouterr().out
    assert rec.ok and rec.text == "ANSWER"
    assert fake_ask.steps() == ["understand", "generate"]
    assert fake_ask.calls[0]["user"] == "focus music"                    # normalized before Gemini
    assert "Parsed profile: {'favorite_genre': 'lofi'" in out
    assert "Backend data (chosen songs):" in out
    assert len(rec.results) == 3 and rec.profile["favorite_genre"] == "lofi"
    assert "Backend data" not in rec.text                                 # terminal only


def test_backend_output_prints_before_the_answer_step_runs(fake_ask, capsys):
    fake_ask.fail = {"generate"}
    rag.get_recommendation_from_text("focus music", k=2)
    assert "Backend data (chosen songs):" in capsys.readouterr().out


def test_text_flow_understanding_failure(fake_ask):
    fake_ask.fail = {"understand"}
    rec = rag.get_recommendation_from_text("focus music")
    assert not rec.ok and rec.text.startswith("Sorry, I couldn't work out your request")
    assert fake_ask.steps() == ["understand"]                             # never reaches retrieval


def test_text_flow_generation_failure_falls_back_to_a_plain_list(fake_ask):
    fake_ask.fail = {"generate"}
    rec = rag.get_recommendation_from_text("focus music", k=3)
    assert not rec.ok
    assert "couldn't write up the explanations" in rec.text
    assert rec.text.count("\n  ") >= 3 and "by " in rec.text             # the 3 songs are listed
    assert "Backend data" not in rec.text
    assert len(rec.results) == 3 and rec.profile


def test_empty_text_makes_no_api_call(fake_ask):
    rec = rag.get_recommendation_from_text("   ")
    assert not rec.ok and "describe what you'd like" in rec.text
    assert fake_ask.calls == []


def test_profile_flow_makes_a_single_gemini_call(fake_ask, capsys):
    name = next(iter(P.PROFILES))
    rec = rag.get_recommendation_from_profile(name, k=4)
    assert rec.ok and rec.text == "ANSWER" and len(rec.results) == 4
    assert fake_ask.steps() == ["generate"]                               # skips query understanding
    request = fake_ask.calls[0]["user"].split("Retrieved songs:")[0]
    assert f"'{name}'" in request and not DIGIT.search(request.split(": ", 1)[1])
    assert "Profile: {" in capsys.readouterr().out


def test_profile_flow_generation_failure(fake_ask):
    fake_ask.fail = {"generate"}
    rec = rag.get_recommendation_from_profile(next(iter(P.PROFILES)))
    assert not rec.ok and "Here are the best matches" in rec.text


def test_unknown_profile_name(fake_ask):
    rec = rag.get_recommendation_from_profile("No Such Profile")
    assert not rec.ok and "don't have a profile" in rec.text
    assert fake_ask.calls == []


def test_adversarial_profiles_are_unreachable_from_rag(fake_ask):
    assert not hasattr(rag, "ADVERSARIAL_PROFILES")
    rec = rag.get_recommendation_from_profile(next(iter(P.ADVERSARIAL_PROFILES)))
    assert not rec.ok and fake_ask.calls == []


def test_profile_flow_accepts_saved_profiles_and_cleans_them(fake_ask):
    saved = {"Loud Night": {"favorite_genre": "edm", "target_energy": 9, "bogus": 1}}
    rec = rag.get_recommendation_from_profile("Loud Night", k=2, profiles=saved)
    assert rec.ok and rec.profile == {"favorite_genre": "edm", "target_energy": 1}


def test_string_wrappers_return_plain_text(fake_ask):
    assert rag.recommend_from_text("focus music") == "ANSWER"
    assert rag.recommend_from_profile(next(iter(P.PROFILES))) == "ANSWER"
    assert isinstance(rag.recommend_from_profile("nope"), str)


# ---------- CLI menu ----------

def test_menu_reprompts_on_invalid_input_then_runs_the_chosen_profile(fake_ask, monkeypatch, capsys):
    answers = iter(["x", "99", "1"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    assert rag.choose_and_recommend() == "ANSWER"
    out = capsys.readouterr().out
    assert out.count("Please enter a number") == 2
    assert fake_ask.steps() == ["generate"]


def test_menu_last_option_asks_for_a_typed_request(fake_ask, monkeypatch):
    last = str(len(P.PROFILES) + 1)
    answers = iter([last, "chill study music"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    assert rag.choose_and_recommend() == "ANSWER"
    assert fake_ask.steps() == ["understand", "generate"]
    assert fake_ask.calls[0]["user"] == "chill study music"


def test_menu_lists_only_user_facing_profiles(fake_ask, monkeypatch, capsys):
    answers = iter(["1"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    rag.choose_and_recommend()
    out = capsys.readouterr().out
    for name in P.PROFILES:
        assert name in out
    for name in P.ADVERSARIAL_PROFILES:
        assert name not in out
