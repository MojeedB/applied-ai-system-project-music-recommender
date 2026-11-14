"""Catalog coverage, genre aliases, dynamic tempo range and similarity tables."""

import itertools

import pytest

import recommender as R


# ---------- genre aliases ----------

@pytest.mark.parametrize("raw, expected", [
    ("lo-fi", "lofi"), ("  Lo-Fi ", "lofi"), ("LOFI", "lofi"), ("lo fi", "lofi"),
    ("hip-hop", "hip hop"), ("rap", "hip hop"), ("rnb", "r&b"),
    ("dnb", "drum and bass"), ("kpop", "k-pop"), ("alt rock", "alternative rock"),
    ("heavy metal", "heavy metal"),      # real catalog genres are left alone
    ("  POP ", "pop"),
])
def test_canonical_genre(raw, expected):
    assert R.canonical_genre(raw) == expected


def test_load_songs_merges_lofi_spellings(tmp_path):
    csv_file = tmp_path / "songs.csv"
    csv_file.write_text(
        "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness\n"
        "1,A,X,lo-fi,focused,0.4,80,0.5,0.6,0.8\n"
        "2,B,X,Lofi,chill,0.3,70,0.5,0.6,0.8\n", encoding="utf-8")
    songs = R.load_songs(str(csv_file))
    assert {s["genre"] for s in songs} == {"lofi"}


def test_user_genre_spelling_matches_catalog_genre():
    song = {"genre": "lofi", "mood": "chill", "energy": 0.4, "tempo_bpm": 80,
            "danceability": 0.6, "acousticness": 0.8}
    exact, _ = R.score_song({"favorite_genre": "lofi"}, song)
    aliased, _ = R.score_song({"favorite_genre": "Lo-Fi"}, song)
    assert exact == aliased == 1.0


# ---------- catalog coverage ----------

def test_every_catalog_mood_has_a_mood_family(catalog):
    missing = {s["mood"] for s in catalog} - set(R.MOOD_FAMILY)
    assert not missing, f"moods without a family: {sorted(missing)}"


def test_every_catalog_genre_has_a_genre_family(catalog):
    """A genre missing here is either a typo in songs.csv (e.g. a mood in the genre column)
    or a new genre that needs an entry in GENRE_FAMILY."""
    missing = {s["genre"] for s in catalog} - set(R.GENRE_FAMILY)
    assert not missing, f"genres without a family: {sorted(missing)}"


def test_new_moods_and_genres_are_present():
    for mood in ("peaceful", "dreamy", "hopeful", "euphoric", "free", "heroic",
                 "triumphant", "determined", "aggressive", "dark"):
        assert mood in R.MOOD_FAMILY
    for genre in ("dream pop", "drum and bass", "neo soul", "heavy metal", "trap", "k-pop"):
        assert genre in R.GENRE_FAMILY


# ---------- similarity tables ----------

def test_override_tables_only_name_known_items():
    mood_names = {n for pair in R.MOOD_OVERRIDE for n in pair}
    genre_names = {n for pair in R.GENRE_OVERRIDE for n in pair}
    assert mood_names <= set(R.MOOD_FAMILY)
    assert genre_names <= set(R.GENRE_FAMILY)


def test_override_tables_have_no_reversed_duplicates():
    for table in (R.MOOD_OVERRIDE, R.GENRE_OVERRIDE):
        for a, b in table:
            assert (b, a) not in table, f"{a!r}/{b!r} listed in both orders"


def test_mood_similarity_is_symmetric_and_in_range():
    for a, b in itertools.product(R.MOOD_FAMILY, repeat=2):
        s = R.mood_sim(a, b)
        assert 0.0 <= s <= 1.0
        assert s == R.mood_sim(b, a)
    assert R.mood_sim("chill", "chill") == 1.0


def test_genre_similarity_is_symmetric_and_in_range():
    for a, b in itertools.product(R.GENRE_FAMILY, repeat=2):
        s = R.genre_sim(a, b)
        assert 0.0 <= s <= 1.0
        assert s == R.genre_sim(b, a)


def test_new_mood_similarities_make_sense():
    assert R.mood_sim("aggressive", "angry") > R.mood_sim("aggressive", "peaceful")
    assert R.mood_sim("peaceful", "chill") >= 0.8
    assert R.mood_sim("heroic", "triumphant") >= 0.8
    assert R.mood_sim("dark", "moody") > R.mood_sim("dark", "happy")
    assert R.mood_sim("happy", "heroic") <= 0.2          # broad family must not over-credit
    assert R.mood_sim("chill", "not-a-mood") == 0.0


def test_new_genre_similarities_make_sense():
    assert R.genre_sim("heavy metal", "metal") >= 0.9
    assert R.genre_sim("heavy metal", "metal") > R.genre_sim("heavy metal", "pop")
    assert R.genre_sim("trance", "edm") > R.genre_sim("trance", "folk")
    assert R.genre_sim("neo soul", "r&b") >= 0.8
    assert R.genre_sim("k-pop", "not-a-genre") == 0.0


# ---------- dynamic tempo range ----------

def test_set_tempo_range_uses_slowest_and_fastest_song():
    R.set_tempo_range([{"tempo_bpm": 90}, {"tempo_bpm": 60}, {"tempo_bpm": 200}])
    assert (R.TEMPO_MIN, R.TEMPO_MAX) == (60, 200)


@pytest.mark.parametrize("songs", [[], [{"tempo_bpm": 100}] * 3])
def test_set_tempo_range_falls_back_for_empty_or_single_tempo_catalog(songs):
    R.set_tempo_range(songs)
    assert (R.TEMPO_MIN, R.TEMPO_MAX) == R.DEFAULT_TEMPO_RANGE
    assert R.TEMPO_MAX > R.TEMPO_MIN          # never a zero-width range (division by zero)


def test_load_songs_sets_range_to_catalog_extremes(catalog):
    tempos = [s["tempo_bpm"] for s in catalog]
    R.set_tempo_range(catalog)
    assert (R.TEMPO_MIN, R.TEMPO_MAX) == (min(tempos), max(tempos))


def test_load_songs_updates_range_when_it_runs(tmp_path):
    csv_file = tmp_path / "songs.csv"
    csv_file.write_text(
        "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness\n"
        "1,A,X,pop,happy,0.4,50,0.5,0.6,0.8\n"
        "2,B,X,pop,happy,0.3,250,0.5,0.6,0.8\n", encoding="utf-8")
    R.load_songs(str(csv_file))
    assert (R.TEMPO_MIN, R.TEMPO_MAX) == (50, 250)


def test_tempo_normalization_follows_the_range_and_clamps():
    R.set_tempo_range([{"tempo_bpm": 100}, {"tempo_bpm": 200}])
    assert R._normalize_tempo(100) == 0.0
    assert R._normalize_tempo(150) == 0.5
    assert R._normalize_tempo(200) == 1.0
    assert R._normalize_tempo(50) == 0.0       # below range
    assert R._normalize_tempo(999) == 1.0      # above range


# ---------- scoring ----------

def test_scores_stay_between_zero_and_one(catalog):
    profiles = [
        {"favorite_genre": "heavy metal", "favorite_mood": "dark", "target_energy": 0.95, "pref_tempo": 180},
        {"favorite_genre": "lofi", "favorite_mood": "peaceful", "target_energy": 0.2},
        {"target_energy": 0.0, "pref_tempo": 0, "target_danceable": 0.0, "target_acoustic": 0.0},
        {"favorite_mood": "euphoric"},
    ]
    for profile in profiles:
        for song in catalog:
            score, _ = R.score_song(profile, song)
            assert 0.0 <= score <= 1.0 + 1e-9


def test_exact_match_ranks_first(catalog):
    profile = {"favorite_genre": "heavy metal", "favorite_mood": "dark",
               "target_energy": 0.97, "pref_tempo": 182}
    top = R.recommend_songs(profile, catalog, k=1)[0][0]
    assert (top["genre"], top["mood"]) == ("heavy metal", "dark")


def test_empty_profile_gives_zero_score(catalog):
    score, reasons = R.score_song({}, catalog[0])
    assert score == 0.0 and reasons == ["no preferences given"]
