"""
Streamlit front end for the music recommender.

Run from the project folder:   streamlit run src/app.py

Two ways to get recommendations (both use the RAG pipeline in rag.py):
  - pick a ready-made profile (built-in profiles plus any you saved)
  - describe what you want in your own words, then optionally save it as a new profile

Successful answers are cached for an hour (shared by all sessions) so repeating a request
doesn't spend free-tier API quota. Failures are never cached. "Regenerate" skips the cache.
The History panel on the right lists the cached answers, titled by the parsed profile.
"""

import os

import streamlit as st

import user_profiles
from result_cache import ResultCache

st.set_page_config(page_title="Musics Match", page_icon="🎵", layout="wide")

# rag.py needs google-genai and python-dotenv; show a clear message instead of a traceback.
try:
    import rag
except ImportError as exc:
    st.error(f"A required package is missing ({exc.name}). "
             "Run `pip install -r requirements.txt` and restart the app.")
    st.stop()


@st.cache_resource
def get_cache() -> ResultCache:
    """One cache for the whole app process (survives Streamlit reruns)."""
    return ResultCache(ttl_seconds=3600, max_size=200)


def get_recommendation(mode: str, key: str, k: int, profiles: dict, force: bool = False):
    """Return (Recommendation, from_cache). Only successful answers are stored."""
    cache = get_cache()
    cache_key = (mode, key, k)

    if not force:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached, True

    if mode == "profile":
        rec = rag.get_recommendation_from_profile(key, k=k, profiles=profiles)
    else:
        rec = rag.get_recommendation_from_text(key, k=k)

    if rec.ok:  # never cache a failure, or an outage would be shown after it recovers
        cache.put(cache_key, rec)
    return rec, False


def show_result(rec, mode: str, key: str, from_cache: bool) -> None:
    """Remember what to display, and what (if anything) can be saved as a profile."""
    st.session_state["answer"] = rec.text.strip()
    st.session_state["from_cache"] = from_cache
    st.session_state["last_request"] = (mode, key)  # what "Regenerate" repeats
    if mode == "text" and rec.ok and rec.profile:
        st.session_state["last_text"] = {"query": key, "profile": rec.profile}
    else:
        st.session_state.pop("last_text", None)


st.title("🎵 Musics Match")
st.caption("Describe what you feel like listening to, or pick a profile from the sidebar.")

if not os.environ.get("GEMINI_API_KEY"):
    st.error(f"GEMINI_API_KEY not found. Add it to `{rag.PROJECT_ROOT / '.env'}` and restart the app "
             "(free key: https://aistudio.google.com/apikey).")
    st.stop()

# Built-in (user-facing) profiles plus the ones users saved. ADVERSARIAL_PROFILES is never included.
saved_profiles = user_profiles.load_user_profiles()
all_profiles = {**rag.PROFILES, **saved_profiles}

# --- Sidebar (left): number of songs first, then one button per profile ---
clicked_profile = None
with st.sidebar:
    k = st.slider("Number of songs", min_value=1, max_value=10, value=5)

    st.subheader("Recommended Profiles")
    last_request = st.session_state.get("last_request")
    for name in all_profiles:
        label = f"⭐ {name}" if name in saved_profiles else name
        is_active = last_request == ("profile", name)
        if st.button(label, key=f"profile_{name}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            clicked_profile = name
    if saved_profiles:
        st.caption("⭐ = profiles you saved")

main_col, history_col = st.columns([3, 1], gap="large")

# ===================== Main column =====================
with main_col:
    flash = st.session_state.pop("flash", None)
    if flash:
        st.success(flash)

    typed = st.text_input("What do you feel like listening to?",
                          placeholder="e.g. upbeat songs for a late-night drive")
    text_key = rag.normalize_query(typed)  # lowercase + collapsed spaces, so equal requests share a cache entry

    col_go, col_regen, _ = st.columns([1, 1, 3])
    go = col_go.button("Recommend", type="primary", key="go")
    regen = col_regen.button("Regenerate", key="regen", disabled="last_request" not in st.session_state,
                             help="Repeat the last request and skip the saved answer")

    # Work out what (if anything) was asked this run: (mode, key, skip_cache)
    request = None
    if clicked_profile:
        request = ("profile", clicked_profile, False)
    elif go:
        if text_key:
            request = ("text", text_key, False)
        else:
            st.warning("Please describe what you'd like to hear first.")
    elif regen and "last_request" in st.session_state:
        mode_, key_ = st.session_state["last_request"]
        request = (mode_, key_, True)

    if request:
        mode, key, force = request
        with st.spinner("Finding songs for you..."):
            rec, from_cache = get_recommendation(mode, key, k, all_profiles, force=force)
        show_result(rec, mode, key, from_cache)
        if (mode, key) != last_request:
            st.rerun()  # the sidebar was drawn before this request ran; redraw to update the highlight

    if st.session_state.get("answer"):
        st.subheader("Your recommendations")
        st.markdown(st.session_state["answer"])
        if st.session_state.get("from_cache"):
            st.caption("Saved answer from the last hour. Press Regenerate for a fresh one.")

    # --- Save the described request as a new profile ---
    last = st.session_state.get("last_text")
    if last and st.session_state.get("answer"):
        st.divider()
        st.markdown("**Like these results? Save your request as a profile**")
        st.caption("Saved profiles appear (with a ⭐) in the sidebar.")
        title = st.text_input("Profile title", max_chars=user_profiles.MAX_TITLE_LEN,
                              placeholder="e.g. Late-night drive",
                              key=f"save_title_{abs(hash(last['query']))}")
        if st.button("Save as profile", key="save"):
            error = user_profiles.save_user_profile(title, last["profile"], reserved=rag.PROFILES)
            if error:
                st.error(error)
            else:
                st.session_state["flash"] = (f"Saved '{' '.join(title.split())}'. "
                                             "Find it in the sidebar.")
                st.session_state.pop("last_text", None)  # avoid saving the same request twice
                st.rerun()

# ===================== History (right) =====================
with history_col:
    st.subheader("History")
    entries = get_cache().entries()  # newest first, expired answers already dropped
    if not entries:
        st.caption("Your recent requests will appear here.")
    for i, (cache_key, rec) in enumerate(entries):
        entry_mode, entry_input, _entry_k = cache_key
        label = rag.short_title(rec.profile or {})  # matches the parsed profile result
        if st.button(label, key=f"hist_{i}", use_container_width=True):
            show_result(rec, entry_mode, entry_input, from_cache=True)
            st.rerun()
        source = f"Profile: {entry_input}" if entry_mode == "profile" else entry_input
        st.caption(source if len(source) <= 40 else source[:37] + "...")
