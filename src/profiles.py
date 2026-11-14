"""
User taste profiles for the music recommender.

Each profile is a dict in the format recommender.score_song() expects.

Two separate groups, defined at the bottom of this file:
  PROFILES              - user-facing profiles (safe to show in menus / pickers)
  ADVERSARIAL_PROFILES  - edge-case test profiles, never user-facing
"""

HIGH_ENERGY_POP = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.90,
    "pref_tempo": 128,
    "target_danceable": 0.85,
    "target_acoustic": 0.15,
}

CHILL_LOFI = {
    "favorite_genre": "lofi",
    "favorite_mood": "chill",
    "target_energy": 0.40,
    "pref_tempo": 80,
    "target_danceable": 0.60,
    "target_acoustic": 0.80,
}

DEEP_INTENSE_ROCK = {
    "favorite_genre": "rock",
    "favorite_mood": "intense",
    "target_energy": 0.92,
    "pref_tempo": 150,
    "target_danceable": 0.65,
    "target_acoustic": 0.08,
}

RAINY_DAY_FOLK = {
    "favorite_genre": "folk",
    "favorite_mood": "sad",
    "target_energy": 0.30,
    "pref_tempo": 84,
    "target_danceable": 0.40,
    "target_acoustic": 0.90,
}

GYM_EDM = {
    "favorite_genre": "edm",
    "favorite_mood": "energetic",
    "target_energy": 0.95,
    "pref_tempo": 128,
    "target_danceable": 0.90,
    "target_acoustic": 0.05,
}

SUNDAY_JAZZ = {
    "favorite_genre": "jazz",
    "favorite_mood": "relaxed",
    "target_energy": 0.37,
    "pref_tempo": 90,
    "target_danceable": 0.55,
    "target_acoustic": 0.88,
}


# --- Adversarial / edge-case profiles ---

# Conflicting signals: features that rarely co-occur in real songs
CONFLICTED_SAD_RAVER = {
    "favorite_genre": "edm",
    "favorite_mood": "melancholy",
    "target_energy": 0.95,
    "pref_tempo": 140,
    "target_danceable": 0.90,
    "target_acoustic": 0.95,
}

# Originally a genre/mood missing from the catalog; k-pop and nostalgic now exist in songs.csv,
# so swap in a genre that is truly absent (e.g. "bossa nova") if you want this case back.
UNKNOWN_GENRE = {
    "favorite_genre": "k-pop",
    "favorite_mood": "nostalgic",
    "target_energy": 0.70,
    "pref_tempo": 110,
    "target_danceable": 0.75,
    "target_acoustic": 0.30,
}

# Extreme boundary values: tests normalization and division by zero
ALL_ZEROS = {
    "favorite_genre": "classical",
    "favorite_mood": "calm",
    "target_energy": 0.0,
    "pref_tempo": 0,
    "target_danceable": 0.0,
    "target_acoustic": 0.0,
}

# Out-of-range values: no input validation
OUT_OF_RANGE = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 1.8,
    "pref_tempo": 400,
    "target_danceable": -0.5,
    "target_acoustic": 0.5,
}

# Wants everything average: near-ties across the catalog
MR_MIDDLE = {
    "favorite_genre": "pop",
    "favorite_mood": "chill",
    "target_energy": 0.5,
    "pref_tempo": 100,
    "target_danceable": 0.5,
    "target_acoustic": 0.5,
}

# Case/whitespace sensitivity in categorical matching
SHOUTY_POP = {
    "favorite_genre": "  POP ",
    "favorite_mood": "Happy",
    "target_energy": 0.8,
    "pref_tempo": 120,
    "target_danceable": 0.8,
    "target_acoustic": 0.2,
}

# Missing keys: tests KeyError handling and default weights
MINIMAL_PROFILE = {
    "favorite_genre": "rock",
}

# Genre loyalist whose audio targets contradict the genre
GENRE_TRAITOR = {
    "favorite_genre": "metal",
    "favorite_mood": "intense",
    "target_energy": 0.05,
    "pref_tempo": 50,
    "target_danceable": 0.10,
    "target_acoustic": 0.95,
}

# --- Groups ---

# User-facing profiles: the only ones that should ever be shown or offered to a user
# (menus, pickers, the default main.py run). Uncomment entries to include more.
PROFILES = {
    "High-Energy Pop": HIGH_ENERGY_POP,
    "Chill Lofi": CHILL_LOFI,
    "Deep Intense Rock": DEEP_INTENSE_ROCK,
    "Rainy Day Folk": RAINY_DAY_FOLK,
    "Gym EMD": GYM_EDM,
    "Sunday Jazz": SUNDAY_JAZZ,
}

# Adversarial / edge-case profiles: for testing and verification only.
# Never expose these to users. Run them with:  python main.py --adversarial
ADVERSARIAL_PROFILES = {
    "Conflicted Sad Raver": CONFLICTED_SAD_RAVER,
    "Unknown Genre": UNKNOWN_GENRE,
    "All Zeros": ALL_ZEROS,
    "Out Of Range": OUT_OF_RANGE,
    "Mr Middle": MR_MIDDLE,
    "Shouty Pop": SHOUTY_POP,
    "Minimal Profile": MINIMAL_PROFILE,
    "Genre Traitor": GENRE_TRAITOR,
}
