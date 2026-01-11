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
    genre_overlap: float = 0.30
    artist_similarity: float = 0.40  # Increased - same artist tracks are highly relevant
    
    # Collaborative weights
    playlist_cooccurrence: float = 0.05
    
    # Popularity calibration
    popularity_match: float = 0.15
    
    # Diversity bonus (prevents too similar tracks)
    diversity_bonus: float = 0.05
    
    # NEW: Release era matching (tracks from similar time periods)
    era_match: float = 0.05
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "audio_similarity": self.audio_similarity,
            "genre_overlap": self.genre_overlap,
            "artist_similarity": self.artist_similarity,
            "playlist_cooccurrence": self.playlist_cooccurrence,
            "popularity_match": self.popularity_match,
            "diversity_bonus": self.diversity_bonus,
            "era_match": self.era_match,
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
    
    # Maximum candidates to consider (increased for better coverage)
    max_candidates: int = 1000
    
    # Minimum playlist co-occurrence threshold
    min_cooccurrence: float = 0.01
    
    # Number of albums to explore per artist (increased)
    albums_per_artist: int = 8
    
    # Number of genre searches to perform
    genre_search_limit: int = 100
    
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
    "pop": ["pop", "dance pop", "electropop", "synth-pop", "indie pop", "art pop", "teen pop", "k-pop", "j-pop", "bedroom pop", "hyperpop"],
    "rock": ["rock", "alternative rock", "indie rock", "classic rock", "hard rock", "punk rock", "post-punk", "new wave", "shoegaze", "grunge", "garage rock", "psychedelic rock", "prog rock", "art rock", "noise rock"],
    "hip_hop": ["hip hop", "rap", "trap", "southern hip hop", "east coast hip hop", "west coast hip hop", "underground hip hop", "alternative hip hop", "experimental hip hop", "conscious hip hop", "gangster rap", "boom bap", "cloud rap", "emo rap", "drill", "grime"],
    "electronic": ["electronic", "edm", "house", "techno", "dubstep", "trance", "drum and bass", "ambient", "idm", "synthwave", "darkwave", "industrial", "breakbeat", "uk garage", "future bass", "lo-fi"],
    "r_and_b": ["r&b", "soul", "neo soul", "contemporary r&b", "funk", "quiet storm", "new jack swing", "alternative r&b"],
    "latin": ["latin", "reggaeton", "latin pop", "salsa", "bachata", "latin hip hop", "latin rock", "corridos", "regional mexican"],
    "country": ["country", "modern country", "country rock", "alt-country", "americana", "bluegrass", "outlaw country"],
    "jazz": ["jazz", "smooth jazz", "jazz fusion", "bebop", "cool jazz", "free jazz", "nu jazz", "acid jazz"],
    "classical": ["classical", "orchestra", "chamber music", "opera", "baroque", "romantic era", "contemporary classical", "minimalism"],
    "metal": ["metal", "heavy metal", "death metal", "black metal", "thrash metal", "doom metal", "progressive metal", "metalcore", "deathcore", "nu metal"],
    "folk": ["folk", "indie folk", "folk rock", "americana", "singer-songwriter", "acoustic", "freak folk"],
    "reggae": ["reggae", "dancehall", "dub", "roots reggae", "ska"],
    "punk": ["punk", "punk rock", "hardcore punk", "pop punk", "post-hardcore", "emo", "screamo"],
    "experimental": ["experimental", "avant-garde", "noise", "drone", "glitch", "musique concrete", "art pop", "dark ambient"],
    "world": ["world", "afrobeat", "afropop", "highlife", "bossa nova", "flamenco", "bollywood"],
}

# Flatten for quick lookup
GENRE_TO_PARENT = {}
for parent, children in GENRE_HIERARCHY.items():
    for child in children:
        GENRE_TO_PARENT[child.lower()] = parent
