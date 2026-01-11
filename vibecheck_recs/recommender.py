"""
Main Recommendation Engine
==========================

Orchestrates the complete recommendation pipeline:
1. Parse playlist input
2. Fetch and extract playlist features
3. Generate candidate tracks
4. Score and rank candidates
5. Apply diversity filtering
6. Generate explanations
7. Return formatted output

This module ties together all components into a cohesive system.
"""

import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

from .spotify_client import SpotifyClient
from .features import (
    FeatureExtractor,
    TrackFeatures,
    PlaylistProfile,
)
from .candidates import CandidateGenerator, DiversityFilter
from .scoring import ScoringEngine, ScoreBreakdown
from .config import (
    NUM_RECOMMENDATIONS,
    DEFAULT_WEIGHTS,
    DEFAULT_CANDIDATE_CONFIG,
    DEFAULT_COLD_START_CONFIG,
    ColdStartConfig,
)


@dataclass
class RecommendationResult:
    """Single track recommendation with explanation."""
    track_id: str
    track_name: str
    artist_names: List[str]
    score: float
    explanation: str
    
    # Optional detailed breakdown
    breakdown: Optional[Dict] = None


@dataclass
class RecommendationOutput:
    """Complete recommendation output."""
    playlist_id: str
    playlist_name: str
    track_count: int
    recommendations: List[RecommendationResult]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "playlist_id": self.playlist_id,
            "playlist_name": self.playlist_name,
            "track_count": self.track_count,
            "recommendations": [
                {
                    "track_id": r.track_id,
                    "track_name": r.track_name,
                    "artist_names": r.artist_names,
                    "score": round(r.score, 4),
                    "explanation": r.explanation,
                }
                for r in self.recommendations
            ]
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class RecommendationEngine:
    """
    Main recommendation engine orchestrating the complete pipeline.
    
    Usage:
        engine = RecommendationEngine()
        result = engine.recommend("spotify:playlist:xxxxx")
        print(result.to_json())
    """
    
    def __init__(
        self,
        spotify_client: Optional[SpotifyClient] = None,
        cold_start_config: ColdStartConfig = DEFAULT_COLD_START_CONFIG
    ):
        """
        Initialize recommendation engine.
        
        Args:
            spotify_client: Pre-configured Spotify client (creates new if None)
            cold_start_config: Cold-start handling configuration
        """
        self.spotify = spotify_client or SpotifyClient()
        self.extractor = FeatureExtractor()
        self.candidate_generator = CandidateGenerator(
            self.spotify,
            self.extractor,
            DEFAULT_CANDIDATE_CONFIG
        )
        self.scorer = ScoringEngine(DEFAULT_WEIGHTS)
        self.diversity_filter = DiversityFilter()
        self.cold_start_config = cold_start_config
    
    def recommend(
        self,
        playlist_input: str,
        n_recommendations: int = NUM_RECOMMENDATIONS,
        simulate_cold_start: bool = False,
        cold_start_ratio: float = 0.5
    ) -> RecommendationOutput:
        """
        Generate recommendations for a playlist.
        
        Args:
            playlist_input: Playlist URL, URI, or ID
            n_recommendations: Number of tracks to recommend
            simulate_cold_start: Whether to simulate cold-start (hide tracks)
            cold_start_ratio: Ratio of tracks to hide in cold-start simulation
            
        Returns:
            RecommendationOutput with ranked recommendations
        """
        print(f"🎵 VibeCheck Recs - Generating recommendations...")
        print("-" * 50)
        
        # Step 1: Fetch playlist data
        print("📥 Fetching playlist data...")
        playlist_id = self.spotify.extract_playlist_id(playlist_input)
        playlist = self.spotify.get_playlist(playlist_id)
        
        playlist_name = playlist.get('name', 'Unknown Playlist')
        all_tracks = playlist.get('all_tracks', [])
        
        print(f"   Playlist: {playlist_name}")
        print(f"   Tracks: {len(all_tracks)}")
        
        # Step 2: Handle cold-start simulation
        if simulate_cold_start:
            print(f"❄️  Simulating cold-start (hiding {cold_start_ratio*100:.0f}% tracks)...")
            visible_tracks = self._simulate_cold_start(all_tracks, cold_start_ratio)
        else:
            visible_tracks = all_tracks
        
        # Step 3: Extract features for playlist tracks
        print("🔬 Extracting playlist features...")
        profile = self._build_playlist_profile(
            playlist_id,
            playlist_name,
            visible_tracks
        )
        
        # Step 4: Generate candidates
        print("🔍 Generating candidate tracks...")
        exclude_ids = set(t['id'] for t in all_tracks if t.get('id'))
        _, candidate_features = self.candidate_generator.generate_candidates(
            profile,
            exclude_ids
        )
        
        print(f"   Candidates with features: {len(candidate_features)}")
        
        if not candidate_features:
            print("⚠️  No candidates found!")
            return RecommendationOutput(
                playlist_id=playlist_id,
                playlist_name=playlist_name,
                track_count=len(all_tracks),
                recommendations=[]
            )
        
        # Step 5: Score candidates
        print("📊 Scoring candidates...")
        scored = self.scorer.score_candidates(
            candidate_features,
            profile,
            return_breakdown=True
        )
        
        # Step 6: Apply diversity filtering
        print("🎲 Applying diversity filter...")
        diverse_results = self.diversity_filter.filter(
            [(tf, score) for tf, score, _ in scored],
            n_select=n_recommendations * 2  # Get extra for safety
        )
        
        # Build final recommendations with explanations
        print("📝 Generating explanations...")
        recommendations = []
        
        # Create mapping from track_id to breakdown
        breakdown_map = {tf.track_id: bd for tf, _, bd in scored if bd}
        
        for tf, score in diverse_results[:n_recommendations]:
            breakdown = breakdown_map.get(tf.track_id)
            explanation = self._generate_explanation(tf, profile, breakdown)
            
            rec = RecommendationResult(
                track_id=tf.track_id,
                track_name=tf.track_name,
                artist_names=tf.artist_names,
                score=score,
                explanation=explanation,
                breakdown=self._format_breakdown(breakdown) if breakdown else None
            )
            recommendations.append(rec)
        
        print("-" * 50)
        print(f"✅ Generated {len(recommendations)} recommendations!")
        
        return RecommendationOutput(
            playlist_id=playlist_id,
            playlist_name=playlist_name,
            track_count=len(all_tracks),
            recommendations=recommendations
        )
    
    def _build_playlist_profile(
        self,
        playlist_id: str,
        playlist_name: str,
        tracks: List[Dict]
    ) -> PlaylistProfile:
        """
        Build playlist profile from tracks.
        
        Args:
            playlist_id: Playlist ID
            playlist_name: Playlist name
            tracks: List of track dictionaries
            
        Returns:
            PlaylistProfile with aggregated features
        """
        # Get track IDs
        track_ids = [t['id'] for t in tracks if t.get('id')]
        
        # Fetch full data
        track_data, audio_features, artists_dict = self.spotify.get_full_track_data(
            track_ids
        )
        
        # Build genre vocabulary from all tracks
        all_genres = []
        for artist in artists_dict.values():
            all_genres.append(artist.get('genres', []))
        self.extractor.build_genre_vocabulary(all_genres)
        
        # Create audio features mapping
        audio_map = {}
        for i, tid in enumerate(track_ids):
            if i < len(audio_features):
                audio_map[tid] = audio_features[i]
        
        # Create track map
        track_map = {t['id']: t for t in track_data if t}
        
        # Extract features for each track
        track_features = []
        for track_id in track_ids:
            if track_id not in track_map:
                continue
            
            track = track_map[track_id]
            audio = audio_map.get(track_id)
            
            tf = self.extractor.extract_track_features(
                track,
                audio,
                artists_dict
            )
            track_features.append(tf)
        
        # Create profile
        profile = self.extractor.create_playlist_profile(
            playlist_id,
            playlist_name,
            track_features
        )
        
        return profile
    
    def _simulate_cold_start(
        self,
        tracks: List[Dict],
        ratio: float
    ) -> List[Dict]:
        """
        Simulate cold-start by hiding a portion of tracks.
        
        Args:
            tracks: All tracks
            ratio: Ratio to hide
            
        Returns:
            Visible tracks after hiding
        """
        import random
        
        n_visible = max(
            self.cold_start_config.min_tracks_for_embedding,
            int(len(tracks) * (1 - ratio))
        )
        
        return random.sample(tracks, min(n_visible, len(tracks)))
    
    def _generate_explanation(
        self,
        track: TrackFeatures,
        profile: PlaylistProfile,
        breakdown: Optional[ScoreBreakdown]
    ) -> str:
        """
        Generate human-readable explanation for a recommendation.
        
        Args:
            track: Recommended track features
            profile: Playlist profile
            breakdown: Score breakdown
            
        Returns:
            Explanation string
        """
        parts = []
        
        # Genre overlap
        if breakdown and breakdown.matched_genres:
            if len(breakdown.matched_genres) > 2:
                genres_str = ", ".join(breakdown.matched_genres[:2]) + f" +{len(breakdown.matched_genres)-2} more"
            else:
                genres_str = ", ".join(breakdown.matched_genres)
            parts.append(f"Genre match: {genres_str}")
        elif track.genres:
            # Check parent genre overlap
            parent_overlap = track.parent_genres & set(profile.parent_genre_distribution.keys())
            if parent_overlap:
                parts.append(f"Similar style: {', '.join(list(parent_overlap)[:2])}")
        
        # Artist connection
        if breakdown and breakdown.artist_overlap:
            parts.append("From an artist in your playlist")
        elif track.artist_names:
            parts.append(f"Artist: {track.artist_names[0]}")
        
        # Audio similarity (only if audio features are available)
        if breakdown and breakdown.audio_score > 0.0:
            if breakdown.audio_score > 0.8:
                parts.append("Very similar vibe and energy")
            elif breakdown.audio_score > 0.6:
                parts.append("Similar sound profile")
        
        # Popularity context
        if abs(track.track_popularity - profile.mean_popularity) < 10:
            parts.append("Matches your popularity preference")
        
        # Fallback
        if not parts:
            parts.append("Matches your playlist's overall vibe")
        
        return "; ".join(parts)
    
    def _format_breakdown(self, breakdown: ScoreBreakdown) -> Dict:
        """Format score breakdown for output."""
        return {
            "audio_similarity": round(breakdown.audio_score, 3),
            "genre_overlap": round(breakdown.genre_score, 3),
            "artist_similarity": round(breakdown.artist_score, 3),
            "popularity_match": round(breakdown.popularity_score, 3),
            "collaborative_score": round(breakdown.collaborative_score, 3),
            "diversity_bonus": round(breakdown.diversity_bonus, 3),
        }


def recommend_from_playlist(
    playlist_url: str,
    n_recommendations: int = 10
) -> Dict:
    """
    Convenience function for quick recommendations.
    
    Args:
        playlist_url: Spotify playlist URL
        n_recommendations: Number of recommendations
        
    Returns:
        Dictionary with recommendations
    """
    engine = RecommendationEngine()
    result = engine.recommend(playlist_url, n_recommendations)
    return result.to_dict()
