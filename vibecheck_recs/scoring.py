"""
Hybrid Scoring Engine
=====================

Computes final scores for candidate tracks using a weighted combination of:
1. Audio similarity (cosine distance to playlist centroid)
2. Genre overlap (hierarchical matching)
3. Artist similarity (direct + related artist matches)
4. Popularity calibration (matching playlist's popularity distribution)
5. Collaborative signals (playlist co-occurrence)

Mathematical Formulation:
-------------------------

Final Score = Σ (w_i × S_i) for each component i

where:
    S_audio = cos(v_candidate, v_centroid)
    S_genre = hierarchical_jaccard(G_candidate, G_playlist)
    S_artist = artist_overlap_score(A_candidate, A_playlist)
    S_pop = 1 - |pop_candidate - mean_pop_playlist| / 100
    S_collab = normalized_cooccurrence_score

Weights are tuned to maximize similarity with Spotify's recommendations.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

from .features import (
    TrackFeatures,
    PlaylistProfile,
    GenreSimilarityCalculator,
)
from .config import (
    ScoringWeights,
    DEFAULT_WEIGHTS,
    AUDIO_FEATURES,
)


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of how a track's score was computed."""
    track_id: str
    final_score: float
    
    # Component scores
    audio_score: float = 0.0
    genre_score: float = 0.0
    artist_score: float = 0.0
    popularity_score: float = 0.0
    collaborative_score: float = 0.0
    diversity_bonus: float = 0.0
    
    # Additional details for explanation
    matched_genres: Optional[List[str]] = None
    audio_feature_diffs: Optional[Dict[str, float]] = None
    artist_overlap: bool = False
    
    def __post_init__(self):
        if self.matched_genres is None:
            self.matched_genres = []
        if self.audio_feature_diffs is None:
            self.audio_feature_diffs = {}


class ScoringEngine:
    """
    Hybrid scoring engine for candidate track ranking.
    
    Combines content-based and collaborative signals to produce
    a final recommendation score.
    """
    
    def __init__(self, weights: ScoringWeights = DEFAULT_WEIGHTS):
        """
        Initialize scoring engine with component weights.
        
        Args:
            weights: Scoring weights configuration
        """
        self.weights = weights
        self.genre_calculator = GenreSimilarityCalculator()
        
        # For cold-start: adaptive weight adjustment
        self.cold_start_mode = False
    
    def score_candidates(
        self,
        candidates: List[TrackFeatures],
        profile: PlaylistProfile,
        return_breakdown: bool = False
    ) -> List[Tuple[TrackFeatures, float, Optional[ScoreBreakdown]]]:
        """
        Score all candidate tracks against the playlist profile.
        
        Args:
            candidates: List of candidate track features
            profile: Playlist profile with aggregated features
            return_breakdown: Whether to return detailed score breakdown
            
        Returns:
            List of (TrackFeatures, score, breakdown) tuples, sorted by score
        """
        results = []
        
        # Precompute some playlist-level data
        playlist_artist_set = profile.unique_artists
        playlist_genre_set = profile.get_all_genres()
        
        # Determine if cold-start mode is needed
        if len(profile.track_features) < 10:
            self.cold_start_mode = True
            adjusted_weights = self._get_cold_start_weights()
        else:
            self.cold_start_mode = False
            adjusted_weights = self.weights
        
        for candidate in candidates:
            breakdown = ScoreBreakdown(
                track_id=candidate.track_id,
                final_score=0.0
            )
            
            # 1. Audio similarity
            audio_score = self._compute_audio_similarity(
                candidate, profile, breakdown
            )
            breakdown.audio_score = audio_score
            
            # 2. Genre overlap
            genre_score = self._compute_genre_similarity(
                candidate, profile, breakdown
            )
            breakdown.genre_score = genre_score
            
            # 3. Artist similarity
            artist_score = self._compute_artist_similarity(
                candidate, profile, playlist_artist_set, breakdown
            )
            breakdown.artist_score = artist_score
            
            # 4. Popularity calibration
            pop_score = self._compute_popularity_match(
                candidate, profile, breakdown
            )
            breakdown.popularity_score = pop_score
            
            # 5. Collaborative signals
            collab_score = self._compute_collaborative_score(
                candidate, breakdown
            )
            breakdown.collaborative_score = collab_score
            
            # 6. Diversity bonus (reward unique contributions)
            diversity_score = self._compute_diversity_bonus(
                candidate, profile, playlist_genre_set, breakdown
            )
            breakdown.diversity_bonus = diversity_score
            
            # Compute weighted final score
            final_score = (
                adjusted_weights.audio_similarity * audio_score +
                adjusted_weights.genre_overlap * genre_score +
                adjusted_weights.artist_similarity * artist_score +
                adjusted_weights.popularity_match * pop_score +
                adjusted_weights.playlist_cooccurrence * collab_score +
                adjusted_weights.diversity_bonus * diversity_score
            )
            
            # Normalize to [0, 1]
            final_score = np.clip(final_score, 0, 1)
            breakdown.final_score = final_score
            
            results.append((
                candidate, 
                final_score, 
                breakdown if return_breakdown else None
            ))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _compute_audio_similarity(
        self,
        candidate: TrackFeatures,
        profile: PlaylistProfile,
        breakdown: ScoreBreakdown
    ) -> float:
        """
        Compute audio feature similarity using cosine similarity.
        
        S_audio = cos(v_candidate, v_centroid)
        
        Note: Audio features API is deprecated for newer Spotify apps.
        Returns neutral score if audio features are unavailable.
        """
        # Check if we have valid audio features (non-zero vectors)
        if profile.audio_centroid is None or len(profile.audio_centroid) == 0:
            return 0.5  # Neutral score if no centroid
        
        # Check if vectors are all zeros (no audio features available)
        if np.allclose(candidate.audio_vector, 0) or np.allclose(profile.audio_centroid, 0):
            return 0.5  # Neutral score - audio features not available
        
        # Reshape for sklearn
        candidate_vec = candidate.audio_vector.reshape(1, -1)
        centroid_vec = profile.audio_centroid.reshape(1, -1)
        
        # Cosine similarity (returns value in [-1, 1], we normalize to [0, 1])
        sim = cosine_similarity(candidate_vec, centroid_vec)[0, 0]
        normalized_sim = (sim + 1) / 2
        
        # Compute per-feature differences for explanation
        diffs = {}
        for i, feature in enumerate(AUDIO_FEATURES):
            if i < len(candidate.audio_vector):
                diff = candidate.audio_vector[i] - profile.audio_centroid[i]
                diffs[feature] = float(diff)
        breakdown.audio_feature_diffs = diffs
        
        return normalized_sim
    
    def _compute_genre_similarity(
        self,
        candidate: TrackFeatures,
        profile: PlaylistProfile,
        breakdown: ScoreBreakdown
    ) -> float:
        """
        Compute genre similarity using hierarchical matching.
        
        S_genre = weighted_jaccard(G_candidate, G_playlist)
        """
        if not candidate.genres or not profile.genre_distribution:
            return 0.0
        
        # Direct overlap with playlist genres (weighted by distribution)
        overlap_score = self.genre_calculator.genre_overlap_with_profile(
            candidate.genres,
            profile
        )
        
        # Hierarchical similarity (parent genre matching)
        profile_genres = list(profile.genre_distribution.keys())
        hierarchical_sim = self.genre_calculator.hierarchical_similarity(
            candidate.genres,
            profile_genres
        )
        
        # Track matched genres for explanation
        candidate_genres_lower = set(g.lower() for g in candidate.genres)
        playlist_genres_lower = set(g.lower() for g in profile.genre_distribution.keys())
        matched = candidate_genres_lower & playlist_genres_lower
        breakdown.matched_genres = list(matched)
        
        # Combine (weight direct overlap more)
        return 0.6 * overlap_score + 0.4 * hierarchical_sim
    
    def _compute_artist_similarity(
        self,
        candidate: TrackFeatures,
        profile: PlaylistProfile,
        playlist_artists: Set[str],
        breakdown: ScoreBreakdown
    ) -> float:
        """
        Compute artist-based similarity.
        
        Considers:
        - Direct artist overlap
        - Artist frequency in playlist (repeated artists are more relevant)
        """
        if not candidate.artist_ids:
            return 0.0
        
        score = 0.0
        has_overlap = False
        
        for artist_id in candidate.artist_ids:
            if artist_id in playlist_artists:
                has_overlap = True
                # Weight by frequency in playlist
                freq = profile.artist_frequency.get(artist_id, 0)
                total = sum(profile.artist_frequency.values())
                if total > 0:
                    score += freq / total
        
        breakdown.artist_overlap = has_overlap
        
        # Cap at 1.0
        return min(score * 2, 1.0)  # Scale up since overlap is rare
    
    def _compute_popularity_match(
        self,
        candidate: TrackFeatures,
        profile: PlaylistProfile,
        breakdown: ScoreBreakdown
    ) -> float:
        """
        Compute popularity calibration score.
        
        S_pop = 1 - |pop_candidate - mean_pop_playlist| / 100
        
        Tracks with similar popularity to playlist average score higher.
        """
        pop_diff = abs(candidate.track_popularity - profile.mean_popularity)
        
        # Consider playlist's popularity variance
        if profile.std_popularity > 0:
            # More lenient if playlist has high variance
            z_score = pop_diff / (profile.std_popularity + 1)
            # Convert to similarity (0 = very different, 1 = very similar)
            score = np.exp(-z_score / 2)
        else:
            score = 1 - (pop_diff / 100)
        
        return max(0, score)
    
    def _compute_collaborative_score(
        self,
        candidate: TrackFeatures,
        breakdown: ScoreBreakdown
    ) -> float:
        """
        Return pre-computed collaborative filtering score.
        
        This score was set during candidate generation based on
        playlist co-occurrence data.
        """
        return candidate.cooccurrence_score
    
    def _compute_diversity_bonus(
        self,
        candidate: TrackFeatures,
        profile: PlaylistProfile,
        playlist_genres: Set[str],
        breakdown: ScoreBreakdown
    ) -> float:
        """
        Compute diversity bonus for tracks that add variety.
        
        Small bonus for:
        - Genres not heavily represented in playlist
        - Related but fresh sounds
        """
        bonus = 0.0
        
        # Genre diversity: reward genres present but not dominant
        candidate_genres = set(g.lower() for g in candidate.genres)
        rare_genres = candidate_genres - playlist_genres
        
        if rare_genres:
            # Bonus for bringing new (but related) genres
            if candidate.parent_genres & set(profile.parent_genre_distribution.keys()):
                bonus += 0.4
            else:
                bonus += 0.1  # Small bonus for completely new genres
        
        # Overlap bonus: tracks with some matching genres get a small bonus too
        if candidate_genres & playlist_genres:
            bonus += 0.3
        
        return min(bonus, 0.8)
    
    def _get_cold_start_weights(self) -> ScoringWeights:
        """
        Adjust weights for cold-start scenarios.
        
        When playlist is small:
        - Reduce collaborative weight (less reliable)
        - Increase genre weight (more generalizable)
        - Increase artist weight (smaller sample = trust direct signals)
        """
        return ScoringWeights(
            audio_similarity=0.0,  # Audio features deprecated
            genre_overlap=0.40,  # Increased
            artist_similarity=0.30,  # Increased
            playlist_cooccurrence=0.05,  # Reduced
            popularity_match=0.15,
            diversity_bonus=0.10,
        )


class BipartiteMatchingEvaluator:
    """
    Evaluates recommendations using optimal bipartite matching.
    
    This simulates the evaluation metric used in the hackathon:
    - Pairwise cosine similarity between recommendations and reference
    - Optimal assignment using Hungarian algorithm
    - Final score = average matched similarity
    """
    
    def compute_matching_score(
        self,
        recommended: List[TrackFeatures],
        reference: List[TrackFeatures]
    ) -> float:
        """
        Compute bipartite matching score between recommended and reference tracks.
        
        Args:
            recommended: Our recommended tracks (as features)
            reference: Ground truth tracks (e.g., Spotify's recommendations)
            
        Returns:
            Average similarity after optimal matching
        """
        if not recommended or not reference:
            return 0.0
        
        n_rec = len(recommended)
        n_ref = len(reference)
        
        # Build feature vectors
        rec_vectors = np.array([
            r.to_combined_vector() for r in recommended
        ])
        ref_vectors = np.array([
            r.to_combined_vector() for r in reference
        ])
        
        # Compute pairwise similarity matrix
        sim_matrix = cosine_similarity(rec_vectors, ref_vectors)
        
        # Convert to cost matrix for Hungarian algorithm
        # (we want to maximize similarity, so use 1 - similarity as cost)
        cost_matrix = 1 - sim_matrix
        
        # Optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Compute average similarity of matched pairs
        matched_similarities = sim_matrix[row_ind, col_ind]
        avg_similarity = np.mean(matched_similarities)
        
        return float(avg_similarity)
    
    def compute_ranking_metrics(
        self,
        recommended_ids: List[str],
        reference_ids: List[str]
    ) -> Dict[str, float]:
        """
        Compute additional ranking metrics.
        
        Args:
            recommended_ids: Our recommended track IDs
            reference_ids: Ground truth track IDs
            
        Returns:
            Dictionary with precision, recall, overlap metrics
        """
        rec_set = set(recommended_ids)
        ref_set = set(reference_ids)
        
        intersection = rec_set & ref_set
        
        precision = len(intersection) / len(rec_set) if rec_set else 0.0
        recall = len(intersection) / len(ref_set) if ref_set else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'overlap_count': len(intersection),
        }
