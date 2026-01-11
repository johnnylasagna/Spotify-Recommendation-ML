"""
Feature Engineering Module
==========================

Extracts and transforms features from tracks, artists, and playlists.
Produces normalized feature vectors suitable for similarity computation.

Feature Categories:
    1. Audio Features (normalized 0-1)
    2. Genre Features (multi-hot encoded + hierarchy)
    3. Popularity Features (z-scored)
    4. Temporal Features (duration-based)
    5. Collaborative Features (playlist co-occurrence)
"""

import numpy as np
from typing import List, Dict, Optional, Set, Tuple, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.sparse import csr_matrix

from .config import (
    AUDIO_FEATURES,
    NORMALIZED_FEATURES,
    FEATURES_TO_NORMALIZE,
    GENRE_HIERARCHY,
    GENRE_TO_PARENT,
)


@dataclass
class TrackFeatures:
    """Complete feature representation of a track."""
    track_id: str
    track_name: str
    
    # Audio features (9 dimensions)
    audio_vector: np.ndarray = field(default_factory=lambda: np.zeros(9))
    
    # Genre features
    genres: List[str] = field(default_factory=list)
    genre_vector: np.ndarray = field(default_factory=lambda: np.zeros(0))
    primary_genre: Optional[str] = None
    parent_genres: Set[str] = field(default_factory=set)
    
    # Popularity features
    track_popularity: float = 0.0
    artist_popularity: float = 0.0
    popularity_gap: float = 0.0  # artist_pop - track_pop
    
    # Temporal features
    duration_ms: int = 0
    duration_minutes: float = 0.0
    duration_z: float = 0.0  # z-scored
    
    # Metadata
    explicit: bool = False
    artist_ids: List[str] = field(default_factory=list)
    artist_names: List[str] = field(default_factory=list)
    album_id: Optional[str] = None
    release_year: Optional[int] = None
    
    # Collaborative features (set during candidate scoring)
    cooccurrence_score: float = 0.0
    
    def to_combined_vector(self, genre_dim: int = 50) -> np.ndarray:
        """
        Combine all features into a single vector for similarity computation.
        
        Returns:
            Combined feature vector
        """
        # Pad or truncate genre vector
        genre_vec = np.zeros(genre_dim)
        if len(self.genre_vector) > 0:
            n = min(len(self.genre_vector), genre_dim)
            genre_vec[:n] = self.genre_vector[:n]
        
        # Combine all features with appropriate scaling
        combined = np.concatenate([
            self.audio_vector,                          # 9 dims
            genre_vec,                                   # genre_dim dims
            [self.track_popularity / 100.0],            # 1 dim (normalized)
            [self.artist_popularity / 100.0],           # 1 dim (normalized)
            [self.duration_minutes / 10.0],             # 1 dim (rough normalization)
            [float(self.explicit)],                      # 1 dim
        ])
        
        return combined


@dataclass 
class PlaylistProfile:
    """Aggregated profile of a playlist's characteristics."""
    playlist_id: str
    playlist_name: str
    
    # Track information
    track_ids: List[str] = field(default_factory=list)
    track_features: List[TrackFeatures] = field(default_factory=list)
    
    # Centroid vectors (mean of all tracks)
    audio_centroid: np.ndarray = field(default_factory=lambda: np.zeros(9))
    genre_centroid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    
    # Aggregate statistics
    mean_popularity: float = 0.0
    std_popularity: float = 0.0
    mean_duration: float = 0.0
    explicit_ratio: float = 0.0
    
    # Genre distribution
    genre_distribution: Dict[str, float] = field(default_factory=dict)
    top_genres: List[str] = field(default_factory=list)
    parent_genre_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Artist distribution
    artist_frequency: Dict[str, int] = field(default_factory=dict)
    unique_artists: Set[str] = field(default_factory=set)
    
    # Temporal profile
    tempo_range: Tuple[float, float] = (0.0, 0.0)
    energy_range: Tuple[float, float] = (0.0, 0.0)
    
    def get_all_genres(self) -> Set[str]:
        """Get all genres present in the playlist."""
        genres = set()
        for tf in self.track_features:
            genres.update(tf.genres)
        return genres


class FeatureExtractor:
    """
    Extracts and normalizes features from Spotify track/artist data.
    """
    
    def __init__(self):
        """Initialize feature extractor with genre vocabulary."""
        self.genre_vocab: Dict[str, int] = {}
        self.genre_idf: Dict[str, float] = {}  # Inverse document frequency
        self._fit_genre_vocab = False
        
        # Scalers for normalization
        self.popularity_scaler = StandardScaler()
        self.duration_scaler = StandardScaler()
        
    def _normalize_audio_feature(self, name: str, value: float) -> float:
        """Normalize a single audio feature to [0, 1]."""
        if name in NORMALIZED_FEATURES:
            return np.clip(value, 0, 1)
        
        if name in FEATURES_TO_NORMALIZE:
            bounds = FEATURES_TO_NORMALIZE[name]
            normalized = (value - bounds['min']) / (bounds['max'] - bounds['min'])
            return np.clip(normalized, 0, 1)
        
        return value
    
    def extract_audio_features(
        self, 
        audio_data: Optional[Dict]
    ) -> np.ndarray:
        """
        Extract normalized audio feature vector.
        
        Args:
            audio_data: Audio features from Spotify API
            
        Returns:
            9-dimensional normalized audio vector
        """
        if not audio_data:
            return np.zeros(len(AUDIO_FEATURES))
        
        vector = []
        for feature in AUDIO_FEATURES:
            value = audio_data.get(feature, 0.0)
            if value is None:
                value = 0.0
            normalized = self._normalize_audio_feature(feature, value)
            vector.append(normalized)
        
        return np.array(vector)
    
    def build_genre_vocabulary(
        self, 
        all_genres: List[List[str]]
    ) -> Dict[str, int]:
        """
        Build genre vocabulary from corpus of tracks.
        
        Args:
            all_genres: List of genre lists from all tracks
            
        Returns:
            Genre to index mapping
        """
        # Count genre frequencies
        genre_counts = Counter()
        for genres in all_genres:
            genre_counts.update([g.lower() for g in genres])
        
        # Create vocabulary (most common first)
        self.genre_vocab = {
            genre: idx 
            for idx, (genre, _) in enumerate(genre_counts.most_common())
        }
        
        # Calculate IDF
        n_docs = len(all_genres)
        for genre, count in genre_counts.items():
            self.genre_idf[genre] = np.log(n_docs / (1 + count))
        
        self._fit_genre_vocab = True
        return self.genre_vocab
    
    def encode_genres(
        self, 
        genres: List[str], 
        use_tfidf: bool = True
    ) -> np.ndarray:
        """
        Encode genres as a vector.
        
        Args:
            genres: List of genre strings
            use_tfidf: Weight by TF-IDF
            
        Returns:
            Genre encoding vector
        """
        if not self._fit_genre_vocab or not self.genre_vocab:
            return np.zeros(1)
        
        vector = np.zeros(len(self.genre_vocab))
        genres_lower = [g.lower() for g in genres]
        
        # Count frequencies in this track
        genre_counts = Counter(genres_lower)
        
        for genre, count in genre_counts.items():
            if genre in self.genre_vocab:
                idx = self.genre_vocab[genre]
                if use_tfidf:
                    tf = count / len(genres_lower) if genres_lower else 0
                    idf = self.genre_idf.get(genre, 1.0)
                    vector[idx] = tf * idf
                else:
                    vector[idx] = 1.0
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def get_parent_genres(self, genres: List[str]) -> Set[str]:
        """Map detailed genres to parent categories."""
        parents = set()
        for genre in genres:
            genre_lower = genre.lower()
            if genre_lower in GENRE_TO_PARENT:
                parents.add(GENRE_TO_PARENT[genre_lower])
            else:
                # Try partial matching
                for child, parent in GENRE_TO_PARENT.items():
                    if child in genre_lower or genre_lower in child:
                        parents.add(parent)
                        break
        return parents
    
    def extract_track_features(
        self,
        track: Dict,
        audio_features: Optional[Dict],
        artist_data: Dict[str, Dict]
    ) -> TrackFeatures:
        """
        Extract complete features for a single track.
        
        Args:
            track: Track metadata from Spotify
            audio_features: Audio features dict
            artist_data: Mapping of artist_id -> artist metadata
            
        Returns:
            TrackFeatures instance
        """
        tf = TrackFeatures(
            track_id=track['id'],
            track_name=track.get('name', ''),
        )
        
        # Audio features
        tf.audio_vector = self.extract_audio_features(audio_features)
        
        # Track metadata
        tf.track_popularity = track.get('popularity', 0)
        tf.duration_ms = track.get('duration_ms', 0)
        tf.duration_minutes = tf.duration_ms / 60000.0
        tf.explicit = track.get('explicit', False)
        
        # Album info
        album = track.get('album', {})
        tf.album_id = album.get('id')
        release_date = album.get('release_date', '')
        if release_date and len(release_date) >= 4:
            try:
                tf.release_year = int(release_date[:4])
            except ValueError:
                pass
        
        # Artist info and genres
        all_genres = []
        artist_pops = []
        
        for artist in track.get('artists', []):
            artist_id = artist.get('id')
            if artist_id:
                tf.artist_ids.append(artist_id)
                tf.artist_names.append(artist.get('name', ''))
                
                if artist_id in artist_data:
                    artist_info = artist_data[artist_id]
                    genres = artist_info.get('genres', [])
                    all_genres.extend(genres)
                    artist_pops.append(artist_info.get('popularity', 0))
        
        tf.genres = list(set(all_genres))
        tf.parent_genres = self.get_parent_genres(tf.genres)
        
        if tf.genres:
            # Primary genre is most specific (longest name typically)
            tf.primary_genre = max(tf.genres, key=len) if tf.genres else None
        
        if artist_pops:
            tf.artist_popularity = float(np.mean(artist_pops))
        tf.popularity_gap = tf.artist_popularity - tf.track_popularity
        
        # Encode genres if vocabulary is built
        if self._fit_genre_vocab:
            tf.genre_vector = self.encode_genres(tf.genres)
        
        return tf
    
    def create_playlist_profile(
        self,
        playlist_id: str,
        playlist_name: str,
        track_features: List[TrackFeatures]
    ) -> PlaylistProfile:
        """
        Create aggregated playlist profile from track features.
        
        Args:
            playlist_id: Spotify playlist ID
            playlist_name: Playlist name
            track_features: List of TrackFeatures for all tracks
            
        Returns:
            PlaylistProfile instance
        """
        profile = PlaylistProfile(
            playlist_id=playlist_id,
            playlist_name=playlist_name,
            track_ids=[tf.track_id for tf in track_features],
            track_features=track_features,
        )
        
        if not track_features:
            return profile
        
        # Compute audio centroid
        audio_vectors = np.array([tf.audio_vector for tf in track_features])
        profile.audio_centroid = np.mean(audio_vectors, axis=0)
        
        # Compute genre distribution
        genre_counts = Counter()
        for tf in track_features:
            genre_counts.update(tf.genres)
        
        total_genres = sum(genre_counts.values())
        if total_genres > 0:
            profile.genre_distribution = {
                g: c / total_genres for g, c in genre_counts.items()
            }
            profile.top_genres = [g for g, _ in genre_counts.most_common(10)]
        
        # Parent genre distribution
        parent_counts = Counter()
        for tf in track_features:
            parent_counts.update(tf.parent_genres)
        total_parents = sum(parent_counts.values())
        if total_parents > 0:
            profile.parent_genre_distribution = {
                g: c / total_parents for g, c in parent_counts.items()
            }
        
        # Genre centroid (average of all genre vectors)
        if track_features[0].genre_vector.size > 0:
            genre_vectors = np.array([
                tf.genre_vector for tf in track_features 
                if tf.genre_vector.size > 0
            ])
            if genre_vectors.size > 0:
                profile.genre_centroid = np.mean(genre_vectors, axis=0)
        
        # Popularity statistics
        popularities = [tf.track_popularity for tf in track_features]
        profile.mean_popularity = float(np.mean(popularities))
        profile.std_popularity = float(np.std(popularities))
        
        # Duration statistics
        durations = [tf.duration_minutes for tf in track_features]
        profile.mean_duration = float(np.mean(durations))
        
        # Explicit ratio
        explicit_count = sum(1 for tf in track_features if tf.explicit)
        profile.explicit_ratio = explicit_count / len(track_features)
        
        # Artist frequency
        artist_counts = Counter()
        for tf in track_features:
            for artist_id in tf.artist_ids:
                artist_counts[artist_id] += 1
        profile.artist_frequency = dict(artist_counts)
        profile.unique_artists = set(artist_counts.keys())
        
        # Audio feature ranges
        if len(audio_vectors) > 0:
            # Tempo is index 8, energy is index 1
            tempos = audio_vectors[:, 8]
            energies = audio_vectors[:, 1]
            profile.tempo_range = (float(np.min(tempos)), float(np.max(tempos)))
            profile.energy_range = (float(np.min(energies)), float(np.max(energies)))
        
        return profile


class GenreSimilarityCalculator:
    """
    Calculates genre similarity using hierarchical matching.
    """
    
    def __init__(self):
        self.hierarchy = GENRE_HIERARCHY
        self.genre_to_parent = GENRE_TO_PARENT
    
    def jaccard_similarity(
        self, 
        genres1: Set[str], 
        genres2: Set[str]
    ) -> float:
        """Jaccard similarity between two genre sets."""
        if not genres1 or not genres2:
            return 0.0
        
        intersection = len(genres1 & genres2)
        union = len(genres1 | genres2)
        
        return intersection / union if union > 0 else 0.0
    
    def hierarchical_similarity(
        self,
        genres1: List[str],
        genres2: List[str]
    ) -> float:
        """
        Compute hierarchical genre similarity.
        
        Gives partial credit for matching parent genres.
        
        Args:
            genres1, genres2: Genre lists to compare
            
        Returns:
            Similarity score [0, 1]
        """
        if not genres1 or not genres2:
            return 0.0
        
        # Exact match score
        set1 = set(g.lower() for g in genres1)
        set2 = set(g.lower() for g in genres2)
        exact_sim = self.jaccard_similarity(set1, set2)
        
        # Parent match score
        parents1 = set()
        parents2 = set()
        for g in genres1:
            g_lower = g.lower()
            if g_lower in self.genre_to_parent:
                parents1.add(self.genre_to_parent[g_lower])
        for g in genres2:
            g_lower = g.lower()
            if g_lower in self.genre_to_parent:
                parents2.add(self.genre_to_parent[g_lower])
        
        parent_sim = self.jaccard_similarity(parents1, parents2)
        
        # Weighted combination (exact match weighted more)
        return 0.7 * exact_sim + 0.3 * parent_sim
    
    def genre_overlap_with_profile(
        self,
        track_genres: List[str],
        profile: PlaylistProfile
    ) -> float:
        """
        Calculate genre overlap between track and playlist profile.
        
        Args:
            track_genres: Track's genres
            profile: Playlist profile
            
        Returns:
            Overlap score [0, 1]
        """
        if not track_genres or not profile.genre_distribution:
            return 0.0
        
        # Weight by playlist genre distribution
        score = 0.0
        track_genres_lower = set(g.lower() for g in track_genres)
        
        for genre in track_genres_lower:
            if genre in profile.genre_distribution:
                score += profile.genre_distribution[genre]
        
        # Normalize
        return min(score, 1.0)


class CollaborativeFeatures:
    """
    Computes collaborative filtering features based on playlist co-occurrence.
    """
    
    def __init__(self):
        # Track -> set of playlist IDs it appears in
        self.track_playlists: Dict[str, Set[str]] = defaultdict(set)
        
        # Track -> track co-occurrence counts
        self.cooccurrence_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        self._built = False
    
    def add_playlist(self, playlist_id: str, track_ids: List[str]):
        """
        Add a playlist to the co-occurrence model.
        
        Args:
            playlist_id: Playlist identifier
            track_ids: Track IDs in the playlist
        """
        for track_id in track_ids:
            self.track_playlists[track_id].add(playlist_id)
        
        # Update co-occurrence
        for i, track_i in enumerate(track_ids):
            for track_j in track_ids[i+1:]:
                self.cooccurrence_matrix[track_i][track_j] += 1
                self.cooccurrence_matrix[track_j][track_i] += 1
        
        self._built = True
    
    def get_cooccurrence_score(
        self,
        candidate_id: str,
        seed_track_ids: Set[str]
    ) -> float:
        """
        Calculate co-occurrence score for a candidate track.
        
        Args:
            candidate_id: Candidate track ID
            seed_track_ids: Set of seed track IDs
            
        Returns:
            Normalized co-occurrence score [0, 1]
        """
        if not self._built or candidate_id not in self.cooccurrence_matrix:
            return 0.0
        
        total_cooc = 0
        for seed_id in seed_track_ids:
            total_cooc += self.cooccurrence_matrix[candidate_id].get(seed_id, 0)
        
        # Normalize by number of seed tracks
        return min(total_cooc / len(seed_track_ids), 1.0) if seed_track_ids else 0.0
    
    def get_playlist_overlap(
        self,
        candidate_id: str,
        seed_track_ids: Set[str]
    ) -> float:
        """
        Calculate playlist overlap between candidate and seed tracks.
        
        Args:
            candidate_id: Candidate track ID
            seed_track_ids: Set of seed track IDs
            
        Returns:
            Jaccard-like overlap score
        """
        if candidate_id not in self.track_playlists:
            return 0.0
        
        candidate_playlists = self.track_playlists[candidate_id]
        
        # Union of playlists containing any seed track
        seed_playlists = set()
        for seed_id in seed_track_ids:
            seed_playlists.update(self.track_playlists.get(seed_id, set()))
        
        if not seed_playlists:
            return 0.0
        
        # Intersection
        overlap = len(candidate_playlists & seed_playlists)
        union = len(candidate_playlists | seed_playlists)
        
        return overlap / union if union > 0 else 0.0
