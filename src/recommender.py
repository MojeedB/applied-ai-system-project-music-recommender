import csv
import heapq
from operator import itemgetter
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    # Every field is optional; a field left as None is skipped when scoring.
    favorite_genre: Optional[str] = None
    favorite_mood: Optional[str] = None
    target_energy: Optional[float] = None
    target_acoustic: Optional[float] = None
    # likes_acoustic: bool
    pref_tempo: Optional[int] = None
    target_danceable: Optional[float] = None


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the k songs that best match the user's profile, highest score first."""
        prefs = asdict(user)
        scored = [(song, *score_song(prefs, asdict(song))) for song in self.songs]
        top = heapq.nlargest(k, scored, key=itemgetter(1))
        return [song for song, _, _ in top]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Explain a song's score as one line, one entry per scored feature."""
        score, reasons = score_song(asdict(user), asdict(song))
        return f"Score {score:.2f}: " + "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    print(f"Loading songs from {csv_path}...")
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_no, row in enumerate(reader, start=2):  # line 1 is the header
            try:
                songs.append({
                    "id": int(row["id"]),
                    "title": row["title"].strip(),
                    "artist": row["artist"].strip(),
                    "genre": canonical_genre(row["genre"]),
                    "mood": row["mood"].strip().lower(),
                    "energy": float(row["energy"]),
                    "tempo_bpm": int(float(row["tempo_bpm"])),
                    "valence": float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                })
            except (KeyError, ValueError, AttributeError) as e:
                raise ValueError(f"{csv_path}, line {line_no}: bad row {row!r} ({e})") from e
    set_tempo_range(songs)
    print(f"Loaded {len(songs)} songs (tempo range {TEMPO_MIN}-{TEMPO_MAX} bpm).")
    return songs

WEIGHTS = {
    "mood": 0.30,
    "genre": 0.15,
    "energy": 0.20,
    "tempo": 0.10,
    "danceability": 0.10,
    "acousticness": 0.15,
}

# Fixed bounds (min/max tempo_bpm in songs.csv) so one song can be scored without the catalog.
# Tempo scaling bounds. These defaults are replaced by the catalog's real min/max whenever
# load_songs() runs (see set_tempo_range); they only matter if a song is scored without loading.
DEFAULT_TEMPO_RANGE = (55, 186)
TEMPO_MIN, TEMPO_MAX = DEFAULT_TEMPO_RANGE


def set_tempo_range(songs: List[Dict]) -> None:
    """Set TEMPO_MIN/TEMPO_MAX to the slowest and fastest tempo in `songs`."""
    global TEMPO_MIN, TEMPO_MAX
    tempos = [s["tempo_bpm"] for s in songs]
    if not tempos or min(tempos) == max(tempos):  # empty or single-tempo catalog: keep a valid span
        TEMPO_MIN, TEMPO_MAX = DEFAULT_TEMPO_RANGE
    else:
        TEMPO_MIN, TEMPO_MAX = min(tempos), max(tempos)

MOOD_FAMILY = {
    "chill": "A", "relaxed": "A", "focused": "A", "peaceful": "A",
    "sad": "B", "melancholic": "B", "soulful": "B",
    "nostalgic": "C", "romantic": "C", "uplifting": "C", "dreamy": "C", "hopeful": "C",
    "happy": "D", "playful": "D", "energetic": "D", "confident": "D",
    "euphoric": "D", "free": "D", "heroic": "D", "triumphant": "D", "determined": "D",
    "intense": "E", "angry": "E", "moody": "E", "aggressive": "E", "dark": "E",
}
MOOD_FAMILY_SIM = {("A", "B"): 0.2, ("A", "C"): 0.3, ("B", "C"): 0.3,
                   ("C", "D"): 0.3, ("D", "E"): 0.2, ("B", "E"): 0.2}
MOOD_OVERRIDE = {
    ("chill", "relaxed"): 0.8, ("chill", "focused"): 0.6,
    ("chill", "happy"): 0.2, ("chill", "moody"): 0.3,
    ("relaxed", "happy"): 0.3, ("relaxed", "moody"): 0.3,
    ("focused", "happy"): 0.1, ("focused", "intense"): 0.2,
    ("focused", "moody"): 0.3, ("happy", "intense"): 0.3,
    ("intense", "moody"): 0.4, ("happy", "moody"): 0.0,
    ("chill", "intense"): 0.0, ("relaxed", "intense"): 0.0,
    ("sad", "melancholic"): 0.8, ("melancholic", "soulful"): 0.4,
    ("nostalgic", "romantic"): 0.5, ("nostalgic", "uplifting"): 0.4,
    ("romantic", "uplifting"): 0.4, ("happy", "playful"): 0.7,
    ("happy", "energetic"): 0.6, ("playful", "energetic"): 0.6,
    ("happy", "uplifting"): 0.6, ("playful", "uplifting"): 0.5,
    ("intense", "angry"): 0.7, ("angry", "moody"): 0.4,
    ("energetic", "intense"): 0.5, ("confident", "intense"): 0.3,
    ("nostalgic", "melancholic"): 0.4, ("moody", "melancholic"): 0.5,
    ("moody", "sad"): 0.4, ("romantic", "soulful"): 0.4,
    ("relaxed", "romantic"): 0.4, ("angry", "sad"): 0.1,
    # peaceful (calm family)
    ("peaceful", "chill"): 0.8, ("peaceful", "relaxed"): 0.8, ("peaceful", "focused"): 0.4,
    # dreamy / hopeful (warm family)
    ("dreamy", "romantic"): 0.6, ("dreamy", "nostalgic"): 0.5, ("dreamy", "chill"): 0.5,
    ("dreamy", "moody"): 0.4, ("dreamy", "uplifting"): 0.4, ("dreamy", "relaxed"): 0.4,
    ("hopeful", "uplifting"): 0.7, ("hopeful", "happy"): 0.5, ("hopeful", "nostalgic"): 0.3,
    ("hopeful", "romantic"): 0.3,
    # euphoric / free / heroic / triumphant / determined (high-spirit family)
    ("euphoric", "energetic"): 0.7, ("euphoric", "happy"): 0.6, ("euphoric", "uplifting"): 0.6,
    ("free", "uplifting"): 0.5, ("free", "happy"): 0.4, ("free", "nostalgic"): 0.3,
    ("heroic", "triumphant"): 0.8, ("heroic", "confident"): 0.5, ("heroic", "determined"): 0.5,
    ("heroic", "intense"): 0.4, ("heroic", "energetic"): 0.4,
    ("triumphant", "confident"): 0.6, ("triumphant", "energetic"): 0.5,
    ("triumphant", "uplifting"): 0.5, ("triumphant", "determined"): 0.5,
    ("determined", "confident"): 0.5, ("determined", "focused"): 0.4,
    ("determined", "intense"): 0.4, ("determined", "energetic"): 0.4,
    # keep the broad "D" family from giving too much credit to unrelated pairs
    ("happy", "heroic"): 0.1, ("happy", "determined"): 0.1, ("happy", "triumphant"): 0.3,
    ("playful", "heroic"): 0.1, ("playful", "determined"): 0.1,
    ("playful", "triumphant"): 0.2, ("free", "heroic"): 0.2, ("free", "determined"): 0.2,
    # aggressive / dark (intense family)
    ("aggressive", "angry"): 0.9, ("aggressive", "intense"): 0.7, ("aggressive", "moody"): 0.3,
    ("dark", "moody"): 0.6, ("dark", "angry"): 0.5, ("dark", "intense"): 0.4,
    ("dark", "melancholic"): 0.3, ("dark", "sad"): 0.3, ("dark", "aggressive"): 0.4,
}

GENRE_FAMILY = {
    "lofi": 1, "jazz": 1, "ambient": 1, "classical": 1,
    "folk": 2, "country": 2, "blues": 2,
    "pop": 3, "indie pop": 3, "r&b": 3,
    "edm": 4, "synthwave": 4,
    "rock": 5, "metal": 5,
    "hip hop": 6, "reggae": 6, "latin": 6,
    # added with the expanded catalog
    "americana": 2, "indie folk": 2,
    "dream pop": 3, "synth pop": 3, "dance pop": 3, "k-pop": 3, "j-pop": 3,
    "latin pop": 3, "neo soul": 3,
    "house": 4, "techno": 4, "trance": 4, "dubstep": 4, "drum and bass": 4,
    "alternative rock": 5, "heavy metal": 5, "doom metal": 5, "power metal": 5,
    "metalcore": 5, "industrial metal": 5,
    "trap": 6, "drill": 6,
}

# Spelling variants -> the canonical genre name used in the catalog.
GENRE_ALIAS = {
    "lo-fi": "lofi", "lo fi": "lofi",
    "hiphop": "hip hop", "hip-hop": "hip hop", "rap": "hip hop",
    "rnb": "r&b", "r and b": "r&b",
    "dnb": "drum and bass", "drum & bass": "drum and bass", "drum n bass": "drum and bass",
    "kpop": "k-pop", "jpop": "j-pop", "synthpop": "synth pop",
    "alt rock": "alternative rock", "alt-rock": "alternative rock",
    "electronic": "edm",
}


def canonical_genre(genre: str) -> str:
    """Lowercase/trim a genre and map spelling variants (e.g. 'lo-fi') to the catalog name."""
    g = genre.strip().lower()
    return GENRE_ALIAS.get(g, g)
GENRE_OVERRIDE = {
    ("pop", "indie pop"): 0.7, ("pop", "r&b"): 0.5,
    ("lofi", "jazz"): 0.6, ("lofi", "ambient"): 0.6,
    ("classical", "ambient"): 0.6, ("classical", "lofi"): 0.3,
    ("folk", "country"): 0.7, ("country", "blues"): 0.5,
    ("synthwave", "edm"): 0.6, ("rock", "metal"): 0.6,
    ("blues", "jazz"): 0.6, ("blues", "rock"): 0.3,
    ("folk", "indie pop"): 0.4, ("country", "pop"): 0.3,
    ("rock", "pop"): 0.3, ("rock", "indie pop"): 0.3,
    ("rock", "synthwave"): 0.2, ("pop", "edm"): 0.4,
    ("pop", "synthwave"): 0.4, ("indie pop", "synthwave"): 0.4,
    ("synthwave", "lofi"): 0.3, ("indie pop", "lofi"): 0.1,
    ("r&b", "hip hop"): 0.6, ("r&b", "jazz"): 0.3,
    ("hip hop", "pop"): 0.3, ("hip hop", "edm"): 0.2,
    ("latin", "pop"): 0.4, ("reggae", "pop"): 0.2,
    # metal sub-genres
    ("heavy metal", "metal"): 0.9, ("power metal", "metal"): 0.7, ("doom metal", "metal"): 0.7,
    ("metalcore", "metal"): 0.7, ("industrial metal", "metal"): 0.7,
    ("heavy metal", "power metal"): 0.7, ("heavy metal", "doom metal"): 0.6,
    ("heavy metal", "metalcore"): 0.7, ("heavy metal", "industrial metal"): 0.6,
    ("power metal", "metalcore"): 0.5, ("doom metal", "industrial metal"): 0.5,
    ("heavy metal", "rock"): 0.5, ("metalcore", "rock"): 0.3,
    ("alternative rock", "rock"): 0.8, ("alternative rock", "metal"): 0.2,
    ("alternative rock", "indie pop"): 0.3,
    # folk / country family
    ("folk", "indie folk"): 0.8, ("folk", "americana"): 0.6, ("country", "americana"): 0.8,
    ("indie folk", "americana"): 0.5, ("indie folk", "country"): 0.4,
    ("indie folk", "indie pop"): 0.4, ("americana", "blues"): 0.4, ("americana", "rock"): 0.3,
    # pop family
    ("dance pop", "pop"): 0.8, ("synth pop", "pop"): 0.7, ("k-pop", "pop"): 0.7,
    ("j-pop", "pop"): 0.7, ("latin pop", "pop"): 0.7, ("dream pop", "pop"): 0.5,
    ("k-pop", "j-pop"): 0.6, ("k-pop", "dance pop"): 0.6, ("j-pop", "dance pop"): 0.5,
    ("synth pop", "dance pop"): 0.6, ("synth pop", "synthwave"): 0.6,
    ("dream pop", "indie pop"): 0.6, ("dream pop", "ambient"): 0.4, ("dream pop", "synthwave"): 0.4,
    ("dream pop", "synth pop"): 0.5, ("latin pop", "latin"): 0.8, ("latin pop", "reggae"): 0.3,
    ("neo soul", "r&b"): 0.8, ("neo soul", "jazz"): 0.4, ("neo soul", "blues"): 0.3,
    ("neo soul", "hip hop"): 0.3,
    # electronic family
    ("house", "edm"): 0.7, ("techno", "edm"): 0.7, ("trance", "edm"): 0.8,
    ("dubstep", "edm"): 0.8, ("drum and bass", "edm"): 0.6, ("dance pop", "edm"): 0.5,
    ("dance pop", "house"): 0.5, ("techno", "house"): 0.7, ("trance", "house"): 0.6,
    ("techno", "trance"): 0.6, ("dubstep", "drum and bass"): 0.5, ("techno", "synthwave"): 0.4,
    ("trance", "synthwave"): 0.3,
    # hip hop family
    ("trap", "hip hop"): 0.8, ("drill", "hip hop"): 0.7, ("drill", "trap"): 0.7,
}


def _pair(table: Dict, a, b):
    """Look up a value for the pair (a, b) in either order, or None if it isn't listed."""
    return table.get((a, b), table.get((b, a)))


def mood_sim(a: str, b: str) -> float:
    """Partial-credit similarity between two moods (1.0 exact, 0.0 unrelated or unknown)."""
    if a == b:
        return 1.0
    s = _pair(MOOD_OVERRIDE, a, b)
    if s is not None:
        return s
    fa, fb = MOOD_FAMILY.get(a), MOOD_FAMILY.get(b)
    if fa is None or fb is None:
        return 0.0
    if fa == fb:
        return 0.5
    return _pair(MOOD_FAMILY_SIM, fa, fb) or 0.0


def genre_sim(a: str, b: str) -> float:
    """Partial-credit similarity between two genres (1.0 exact, 0.0 unrelated or unknown)."""
    if a == b:
        return 1.0
    s = _pair(GENRE_OVERRIDE, a, b)
    if s is not None:
        return s
    fa, fb = GENRE_FAMILY.get(a), GENRE_FAMILY.get(b)
    return 0.4 if fa is not None and fa == fb else 0.0


def _closeness(song_value: float, target: float) -> float:
    """Return 1.0 when a song's value equals the target, falling toward 0 as they differ."""
    return 1.0 - abs(song_value - target)


def _normalize_tempo(bpm: float) -> float:
    """Scale a tempo in BPM to 0-1 using TEMPO_MIN and TEMPO_MAX, clamping out-of-range values."""
    return min(max((bpm - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN), 0.0), 1.0)


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py

    user_prefs keys (all optional; missing or None fields are skipped and the
    remaining weights are rescaled so the score stays between 0 and 1):
      favorite_mood, favorite_genre, target_energy, pref_tempo,
      target_danceable, target_acoustic
    Returns (score, reasons), where each reason shows one feature's contribution.
    """
    # (feature, preference label, song label, similarity)
    terms: List[Tuple[str, str, str, float]] = []

    mood = user_prefs.get("favorite_mood")
    if mood and mood.strip():  # None or a blank string means "no preference"
        mood = mood.strip().lower()
        terms.append(("mood", mood, song["mood"], mood_sim(mood, song["mood"])))

    genre = user_prefs.get("favorite_genre")
    if genre and genre.strip():
        genre = canonical_genre(genre)
        terms.append(("genre", genre, song["genre"], genre_sim(genre, song["genre"])))

    energy = user_prefs.get("target_energy")
    if energy is not None:
        terms.append(("energy", f"{energy:.2f}", f"{song['energy']:.2f}",
                      _closeness(song["energy"], energy)))

    tempo = user_prefs.get("pref_tempo")
    if tempo is not None:
        terms.append(("tempo", f"{tempo} bpm", f"{song['tempo_bpm']} bpm",
                      _closeness(_normalize_tempo(song["tempo_bpm"]), _normalize_tempo(tempo))))

    dance = user_prefs.get("target_danceable")
    if dance is not None:
        terms.append(("danceability", f"{dance:.2f}", f"{song['danceability']:.2f}",
                      _closeness(song["danceability"], dance)))

    acoustic = user_prefs.get("target_acoustic")
    if acoustic is not None:
        terms.append(("acousticness", f"{acoustic:.2f}", f"{song['acousticness']:.2f}",
                      _closeness(song["acousticness"], acoustic)))

    total_weight = sum(WEIGHTS[name] for name, _, _, _ in terms)
    if total_weight == 0:
        return 0.0, ["no preferences given"]

    score = 0.0
    reasons: List[str] = []
    for name, wanted, actual, sim in terms:
        points = WEIGHTS[name] * sim / total_weight
        score += points
        reasons.append(f"{name}: wanted {wanted}, got {actual} (similarity {sim:.2f}, +{points:.2f})")
    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    """
    # Previous loop-based version, kept for reference:
    # scored = []
    # for song in songs:
    #     score, reasons = score_song(user_prefs, song)
    #     scored.append((song, score, "; ".join(reasons)))
    # # Highest score first; ties fall back to song id so the order is stable.
    # scored.sort(key=lambda item: (-item[1], item[0]["id"]))
    # return scored[:k]

    scored = [(song, *score_song(user_prefs, song)) for song in songs]  # (song, score, reasons)
    # nlargest keeps equal scores in input order, so ties resolve by song order.
    top = heapq.nlargest(k, scored, key=itemgetter(1))
    return [(song, score, "; ".join(reasons)) for song, score, reasons in top]
