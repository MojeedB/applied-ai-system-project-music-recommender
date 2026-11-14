"""Streamlit app behavior, driven headlessly with Streamlit's AppTest. Gemini is faked."""

import re
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

import profiles as P  # noqa: E402
import rag  # noqa: E402
import user_profiles  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "src" / "app.py")


class FakeAsk:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def __call__(self, system, user, json_mode=False, **kwargs):
        self.calls += 1
        if self.fail:
            raise rag.GeminiError("The AI service is getting a lot of requests.")
        if json_mode:
            return '{"favorite_genre": "lofi", "favorite_mood": "focused", "target_energy": 0.3}'
        return f"ANSWER#{self.calls}"


@pytest.fixture
def fake(monkeypatch, tmp_path):
    f = FakeAsk()
    monkeypatch.setattr(rag, "_ask", f)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(user_profiles, "USER_PROFILES_PATH", tmp_path / "user_profiles.json")
    st.cache_resource.clear()                      # fresh shared cache for every test
    yield f
    st.cache_resource.clear()


def new_app():
    return AppTest.from_file(APP, default_timeout=30).run()


def answers(at):
    return [m.value for m in at.markdown if m.value.startswith("ANSWER")]


def history_labels(at):
    return [b.label for b in at.button if b.key and b.key.startswith("hist_")]


def saved_notice(at):
    return [c.value for c in at.caption if "Saved answer" in c.value]


FIRST_PROFILE = next(iter(P.PROFILES))


# ---------- startup ----------

def test_missing_api_key_shows_a_clear_error_instead_of_crashing(monkeypatch, fake):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    at = new_app()
    assert not at.exception
    assert any("GEMINI_API_KEY" in e.value for e in at.error)


def test_layout_slider_first_then_profile_buttons_and_text_box_on_main_page(fake):
    at = new_app()
    assert not at.exception
    kinds = [type(e).__name__ for e in at.sidebar]
    assert kinds.index("Slider") < kinds.index("Button")             # slider at the top of the sidebar
    assert [b.label for b in at.sidebar.button] == list(P.PROFILES)  # one button per profile
    assert not at.selectbox and not at.radio                         # dropdown and mode switch are gone
    assert [t.label for t in at.text_input] == ["What do you feel like listening to?"]


def test_adversarial_profiles_are_never_shown(fake):
    labels = " ".join(b.label for b in at_buttons(new_app()))
    for name in P.ADVERSARIAL_PROFILES:
        assert name not in labels


def at_buttons(at):
    return list(at.sidebar.button) + list(at.button)


# ---------- profile buttons ----------

def test_clicking_a_profile_button_shows_an_answer_and_highlights_it(fake):
    at = new_app()
    at.sidebar.button(key=f"profile_{FIRST_PROFILE}").click().run()
    assert not at.exception
    assert answers(at) == ["ANSWER#1"] and fake.calls == 1          # profile route: one Gemini call
    assert [b.label for b in at.sidebar.button if b.proto.type == "primary"] == [FIRST_PROFILE]


def test_second_click_on_the_same_profile_uses_the_cache(fake):
    at = new_app()
    btn = f"profile_{FIRST_PROFILE}"
    at.sidebar.button(key=btn).click().run()
    at.sidebar.button(key=btn).click().run()
    assert fake.calls == 1 and answers(at) == ["ANSWER#1"]
    assert saved_notice(at)


def test_regenerate_repeats_the_last_request_and_skips_the_cache(fake):
    at = new_app()
    assert at.button(key="regen").disabled                           # nothing to repeat yet
    at.sidebar.button(key=f"profile_{FIRST_PROFILE}").click().run()
    at.button(key="regen").click().run()
    assert fake.calls == 2 and answers(at) == ["ANSWER#2"]
    assert not saved_notice(at)


# ---------- typed requests ----------

def test_typed_request_runs_both_gemini_steps_and_offers_to_save(fake):
    at = new_app()
    at.text_input[0].set_value("Study music")
    at.button(key="go").click().run()
    assert not at.exception and fake.calls == 2
    assert answers(at) == ["ANSWER#2"]
    assert any(b.key == "save" for b in at.button)                   # save form appears for text requests


def test_differently_typed_equal_requests_share_one_cache_entry(fake):
    at = new_app()
    at.text_input[0].set_value("Chill Music")
    at.button(key="go").click().run()
    at.text_input[0].set_value("  chill   MUSIC ")
    at.button(key="go").click().run()
    assert fake.calls == 2                                           # only the first one hit Gemini
    assert saved_notice(at)


def test_regenerate_after_a_typed_request_repeats_it(fake):
    at = new_app()
    at.text_input[0].set_value("Study music")
    at.button(key="go").click().run()
    at.button(key="regen").click().run()
    assert fake.calls == 4


def test_empty_text_shows_a_warning_and_makes_no_api_call(fake):
    at = new_app()
    at.text_input[0].set_value("   ")
    at.button(key="go").click().run()
    assert any("describe what you'd like" in w.value for w in at.warning)
    assert fake.calls == 0


def test_failures_are_shown_but_not_cached(fake):
    at = new_app()
    at.text_input[0].set_value("jazz")
    fake.fail = True
    at.button(key="go").click().run()
    assert any(m.value.startswith("Sorry") for m in at.markdown)
    assert history_labels(at) == []                                  # nothing cached or listed
    fake.fail = False
    at.button(key="go").click().run()
    assert answers(at) and not saved_notice(at)                      # fresh answer after recovery


# ---------- history ----------

def test_history_lists_cached_answers_newest_first_with_parsed_profile_titles(fake):
    at = new_app()
    assert any("will appear here" in c.value for c in at.caption)
    at.text_input[0].set_value("study music")
    at.button(key="go").click().run()
    at.sidebar.button(key=f"profile_{FIRST_PROFILE}").click().run()
    labels = history_labels(at)
    assert len(labels) == 2
    assert labels[1] == "Lofi · Focused · Calm"                       # parsed profile title (older)
    assert labels[0] == rag.short_title(rag.clean_profile(P.PROFILES[FIRST_PROFILE]))


def test_clicking_a_history_entry_restores_its_answer_without_an_api_call(fake):
    at = new_app()
    at.sidebar.button(key=f"profile_{FIRST_PROFILE}").click().run()
    at.text_input[0].set_value("study music")
    at.button(key="go").click().run()
    calls = fake.calls
    oldest = [b for b in at.button if b.key and b.key.startswith("hist_")][-1]
    oldest.click().run()
    assert fake.calls == calls
    assert answers(at) == ["ANSWER#1"] and saved_notice(at)


# ---------- saving a described request as a profile ----------

def save_with_title(at, title):
    box = next(t for t in at.text_input if t.label == "Profile title")
    box.set_value(title)
    at.button(key="save").click().run()


def test_save_form_only_appears_after_a_typed_request(fake):
    at = new_app()
    at.sidebar.button(key=f"profile_{FIRST_PROFILE}").click().run()
    assert not any(b.key == "save" for b in at.button)


def test_saving_requires_a_title(fake):
    at = new_app()
    at.text_input[0].set_value("study music")
    at.button(key="go").click().run()
    save_with_title(at, "")
    assert any("enter a title" in e.value for e in at.error)
    assert user_profiles.load_user_profiles() == {}


def test_saved_profile_persists_and_shows_up_as_a_starred_sidebar_button(fake):
    at = new_app()
    at.text_input[0].set_value("study music")
    at.button(key="go").click().run()
    save_with_title(at, "Study  Time")
    assert list(user_profiles.load_user_profiles()) == ["Study Time"]
    assert "last_text" not in at.session_state                       # can't be saved twice

    fresh = new_app()                                                # a brand-new session
    labels = [b.label for b in fresh.sidebar.button]
    assert labels[-1] == "⭐ Study Time"
    fresh.sidebar.button(key="profile_Study Time").click().run()
    assert not fresh.exception and answers(fresh)
    assert fake.calls == 3                                           # 2 for the typed request + 1 here


def test_duplicate_and_reserved_titles_are_rejected(fake):
    at = new_app()
    at.text_input[0].set_value("study music")
    at.button(key="go").click().run()
    save_with_title(at, FIRST_PROFILE.lower())                       # clashes with a built-in
    assert any("already exists" in e.value for e in at.error)
    assert user_profiles.load_user_profiles() == {}
