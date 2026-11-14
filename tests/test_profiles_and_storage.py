"""profiles.py groups, user_profiles.py persistence and result_cache.py."""

import json

import pytest

import profiles as P
import recommender as R
import user_profiles as U
from result_cache import ResultCache


# ---------- profiles.py groups ----------

def test_adversarial_profiles_are_not_user_facing():
    user_ids = {id(p) for p in P.PROFILES.values()}
    for name, profile in P.ADVERSARIAL_PROFILES.items():
        assert name not in P.PROFILES
        assert id(profile) not in user_ids


def test_both_groups_are_non_empty_and_disjoint_by_name():
    assert P.PROFILES and P.ADVERSARIAL_PROFILES
    assert not set(P.PROFILES) & set(P.ADVERSARIAL_PROFILES)


def test_user_facing_profiles_only_use_catalog_vocabulary(catalog):
    genres = {s["genre"] for s in catalog}
    moods = {s["mood"] for s in catalog}
    for name, profile in P.PROFILES.items():
        if "favorite_genre" in profile:
            assert R.canonical_genre(profile["favorite_genre"]) in genres, name
        if "favorite_mood" in profile:
            assert profile["favorite_mood"].strip().lower() in moods, name


def test_every_profile_runs_through_the_recommender(catalog):
    for group in (P.PROFILES, P.ADVERSARIAL_PROFILES):
        for name, profile in group.items():
            assert len(R.recommend_songs(profile, catalog, k=5)) == 5, name


# ---------- user_profiles.py ----------

def test_default_path_is_inside_the_data_folder():
    assert U.USER_PROFILES_PATH.parent.name == "data"
    assert U.USER_PROFILES_PATH.name == "user_profiles.json"


def test_load_returns_empty_for_missing_corrupt_or_wrong_shape(tmp_path):
    path = tmp_path / "up.json"
    assert U.load_user_profiles(path) == {}
    path.write_text("not json", encoding="utf-8")
    assert U.load_user_profiles(path) == {}
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert U.load_user_profiles(path) == {}


def test_load_skips_entries_that_are_not_profiles(tmp_path):
    path = tmp_path / "up.json"
    path.write_text(json.dumps({"Good": {"favorite_genre": "pop"}, "Bad": "oops", "Worse": 5}),
                    encoding="utf-8")
    assert U.load_user_profiles(path) == {"Good": {"favorite_genre": "pop"}}


def test_save_then_load_round_trip(tmp_path):
    path = tmp_path / "nested" / "up.json"          # parent folder is created
    profile = {"favorite_genre": "synthwave", "target_energy": 0.7}
    assert U.save_user_profile("Night Drive", profile, path=path) is None
    assert U.load_user_profiles(path) == {"Night Drive": profile}
    assert not path.with_suffix(".tmp").exists()    # temp file cleaned up by the atomic replace


def test_saving_a_second_profile_keeps_the_first(tmp_path):
    path = tmp_path / "up.json"
    U.save_user_profile("One", {"favorite_genre": "pop"}, path=path)
    U.save_user_profile("Two", {"favorite_genre": "jazz"}, path=path)
    assert set(U.load_user_profiles(path)) == {"One", "Two"}


@pytest.mark.parametrize("title, expected_fragment", [
    ("", "enter a title"),
    ("   ", "enter a title"),
    ("x" * (U.MAX_TITLE_LEN + 1), "characters or fewer"),
])
def test_invalid_titles_are_rejected(tmp_path, title, expected_fragment):
    error = U.save_user_profile(title, {"favorite_genre": "pop"}, path=tmp_path / "up.json")
    assert error and expected_fragment in error
    assert not (tmp_path / "up.json").exists()


def test_title_at_the_length_limit_is_accepted(tmp_path):
    title = "x" * U.MAX_TITLE_LEN
    assert U.save_user_profile(title, {"favorite_genre": "pop"}, path=tmp_path / "up.json") is None


def test_duplicate_titles_rejected_ignoring_case_and_spacing(tmp_path):
    path = tmp_path / "up.json"
    assert U.save_user_profile("Late  Night", {"favorite_genre": "pop"}, path=path) is None
    error = U.save_user_profile("  late night ", {"favorite_genre": "jazz"}, path=path)
    assert error and "already exists" in error
    assert U.load_user_profiles(path)["Late Night"] == {"favorite_genre": "pop"}   # not overwritten


def test_titles_of_built_in_profiles_are_reserved(tmp_path):
    error = U.save_user_profile("chill lofi", {"favorite_genre": "pop"},
                                reserved=["Chill Lofi"], path=tmp_path / "up.json")
    assert error and "already exists" in error


def test_title_whitespace_is_collapsed_when_saved(tmp_path):
    path = tmp_path / "up.json"
    U.save_user_profile("  Study    Time ", {"favorite_genre": "lofi"}, path=path)
    assert list(U.load_user_profiles(path)) == ["Study Time"]


def test_empty_profile_is_not_saved(tmp_path):
    error = U.save_user_profile("Nothing", {}, path=tmp_path / "up.json")
    assert error and "didn't produce" in error


def test_unwritable_location_returns_an_error_instead_of_raising(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder")
    error = U.save_user_profile("X", {"favorite_genre": "pop"}, path=blocker / "up.json")
    assert error and "couldn't save" in error


# ---------- result_cache.py ----------

class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_cache(ttl=10, max_size=3):
    clock = Clock()
    return ResultCache(ttl_seconds=ttl, max_size=max_size, clock=clock), clock


def test_cache_miss_returns_none():
    cache, _ = make_cache()
    assert cache.get("nope") is None


def test_cache_returns_stored_value_until_ttl_passes():
    cache, clock = make_cache(ttl=10)
    cache.put("a", 1)
    clock.now = 10                       # exactly at the limit: still valid
    assert cache.get("a") == 1
    clock.now = 10.01
    assert cache.get("a") is None
    assert len(cache) == 0               # expired entry was removed


def test_cache_evicts_the_oldest_entry_when_full():
    cache, _ = make_cache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2 and cache.get("c") == 3


def test_putting_an_existing_key_refreshes_it():
    cache, clock = make_cache(ttl=10, max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    clock.now = 5
    cache.put("a", 99)                   # refreshed: newest and timestamp reset
    cache.put("c", 3)                    # evicts "b" (now the oldest), not "a"
    assert cache.get("b") is None
    assert cache.get("a") == 99
    clock.now = 14                       # 9s after the refresh, but 14s after the original put
    assert cache.get("a") == 99


def test_entries_are_newest_first_and_skip_expired():
    cache, clock = make_cache(ttl=10, max_size=5)
    cache.put("a", 1)
    clock.now = 1
    cache.put("b", 2)
    clock.now = 2
    cache.put("c", 3)
    assert cache.entries() == [("c", 3), ("b", 2), ("a", 1)]
    clock.now = 10.5                     # only "a" (stored at 0) has expired
    assert [k for k, _ in cache.entries()] == ["c", "b"]


def test_cache_clear():
    cache, _ = make_cache()
    cache.put("a", 1)
    cache.clear()
    assert len(cache) == 0 and cache.entries() == []


def test_cache_accepts_tuple_keys_like_the_app_uses():
    cache, _ = make_cache()
    cache.put(("text", "chill music", 5), "answer")
    assert cache.get(("text", "chill music", 5)) == "answer"
    assert cache.get(("text", "chill music", 6)) is None
