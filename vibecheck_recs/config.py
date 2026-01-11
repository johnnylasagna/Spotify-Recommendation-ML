"""
Configuration and constants for VibeCheck Recs recommendation system.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict

# =============================================================================
# SPOTIFY API CONFIGURATION
# =============================================================================
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.environ.get("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

# =============================================================================
# AUDIO FEATURE CONFIGURATION
# =============================================================================
AUDIO_FEATURES = [
    "danceability",
    "energy", 
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# Features that are already normalized [0, 1]
NORMALIZED_FEATURES = [
    "danceability",
    "energy",
    "speechiness", 
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]

# Features that need normalization
FEATURES_TO_NORMALIZE = {
    "loudness": {"min": -60, "max": 0},  # dB
    "tempo": {"min": 50, "max": 200},     # BPM
}

# =============================================================================
# SCORING WEIGHTS (tuned for Spotify similarity)
# =============================================================================
@dataclass
class ScoringWeights:
    """Weights for the hybrid scoring function."""
    # Content-based weights (adjusted for no audio features API)
    audio_similarity: float = 0.0  # Deprecated by Spotify
    genre_overlap: float = 0.35
    artist_similarity: float = 0.25
    
    # Collaborative weights
    playlist_cooccurrence: float = 0.10
    
    # Popularity calibration
    popularity_match: float = 0.20
    
    # Diversity bonus (prevents too similar tracks)
    diversity_bonus: float = 0.10
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "audio_similarity": self.audio_similarity,
            "genre_overlap": self.genre_overlap,
            "artist_similarity": self.artist_similarity,
            "playlist_cooccurrence": self.playlist_cooccurrence,
            "popularity_match": self.popularity_match,
            "diversity_bonus": self.diversity_bonus,
        }

DEFAULT_WEIGHTS = ScoringWeights()

# =============================================================================
# CANDIDATE GENERATION CONFIGURATION
# =============================================================================
@dataclass
class CandidateConfig:
    """Configuration for candidate track generation."""
    # Number of related artists to explore per seed artist (deprecated API)
    related_artists_per_seed: int = 5
    
    # Number of top tracks to fetch per artist
    top_tracks_per_artist: int = 10
    
    # Maximum candidates to consider
    max_candidates: int = 500
    
    # Minimum playlist co-occurrence threshold
    min_cooccurrence: float = 0.01
    
    # Number of albums to explore per artist
    albums_per_artist: int = 5
    
    # Number of genre searches to perform
    genre_search_limit: int = 50
    
    # Use search-based candidate discovery (fallback when related artists unavailable)
    use_search_fallback: bool = True

DEFAULT_CANDIDATE_CONFIG = CandidateConfig()

# =============================================================================
# COLD-START HANDLING
# =============================================================================
@dataclass
class ColdStartConfig:
    """Configuration for cold-start robustness."""
    # Minimum tracks needed for reliable embedding
    min_tracks_for_embedding: int = 5
    
    # Fallback to genre-based when tracks < threshold
    genre_fallback_threshold: int = 10
    
    # Bootstrap multiplier for small playlists
    bootstrap_multiplier: float = 1.5
    
    # Use artist expansion when playlist is sparse
    use_artist_expansion: bool = True

DEFAULT_COLD_START_CONFIG = ColdStartConfig()

# =============================================================================
# CACHING CONFIGURATION
# =============================================================================
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
CACHE_TTL_HOURS = 24  # Cache time-to-live

# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================
NUM_RECOMMENDATIONS = 10
OUTPUT_FORMAT = "json"  # json or csv

# =============================================================================
# GENRE TAXONOMY (hierarchical grouping for better matching)
# =============================================================================
GENRE_HIERARCHY = {
    "pop": ["pop", "dance pop", "electropop", "synth-pop", "indie pop", "art pop"],
    "rock": ["rock", "alternative rock", "indie rock", "classic rock", "hard rock", "punk rock"],
    "hip_hop": ["hip hop", "rap", "trap", "southern hip hop", "east coast hip hop", "west coast hip hop"],
    "electronic": ["electronic", "edm", "house", "techno", "dubstep", "trance", "drum and bass"],
    "r_and_b": ["r&b", "soul", "neo soul", "contemporary r&b", "funk"],
    "latin": ["latin", "reggaeton", "latin pop", "salsa", "bachata", "latin hip hop"],
    "country": ["country", "modern country", "country rock", "alt-country"],
    "jazz": ["jazz", "smooth jazz", "jazz fusion", "bebop", "cool jazz"],
    "classical": ["classical", "orchestra", "chamber music", "opera", "baroque"],
    "metal": ["metal", "heavy metal", "death metal", "black metal", "thrash metal"],
    "folk": ["folk", "indie folk", "folk rock", "americana", "singer-songwriter"],
    "reggae": ["reggae", "dancehall", "dub", "roots reggae"],
}

# Flatten for quick lookup
GENRE_TO_PARENT = {}
for parent, children in GENRE_HIERARCHY.items():
    for child in children:
        GENRE_TO_PARENT[child.lower()] = parent
