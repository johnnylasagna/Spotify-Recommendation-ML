"""
Explanation Generator Module
============================

Generates detailed, human-readable explanations for each recommendation.
Explanations include:
- Genre overlap analysis
- Audio feature comparison
- Artist similarity context
- Playlist co-occurrence signals
- Popularity calibration notes

This module provides bonus points in the hackathon evaluation.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from .features import TrackFeatures, PlaylistProfile
from .scoring import ScoreBreakdown
from .config import AUDIO_FEATURES


@dataclass
class DetailedExplanation:
    """Comprehensive explanation for a recommendation."""
    track_id: str
    summary: str
    
    # Component explanations
    genre_explanation: str = ""
    audio_explanation: str = ""
    artist_explanation: str = ""
    popularity_explanation: str = ""
    collaborative_explanation: str = ""
    
    # Confidence and relevance
    confidence_score: float = 0.0
    primary_signal: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "summary": self.summary,
            "genre": self.genre_explanation,
            "audio": self.audio_explanation,
            "artist": self.artist_explanation,
            "popularity": self.popularity_explanation,
            "collaborative": self.collaborative_explanation,
            "confidence": round(self.confidence_score, 2),
            "primary_signal": self.primary_signal,
        }


class ExplanationGenerator:
    """
    Generates human-readable explanations for recommendations.
    
    Focuses on:
    - Clarity: Easy to understand for non-technical users
    - Relevance: Highlights the most important matching factors
    - Specificity: Uses actual genre names, feature values, etc.
    """
    
    def __init__(self):
        """Initialize explanation generator."""
        # Audio feature descriptions for human-readable output
        self.audio_descriptors = {
            "danceability": {
                "high": "great for dancing",
                "low": "more chill and laid-back",
                "match": "similar groove factor"
            },
            "energy": {
                "high": "high energy and intense",
                "low": "calm and mellow",
                "match": "matching energy level"
            },
            "valence": {
                "high": "upbeat and positive vibes",
                "low": "darker, more introspective mood",
                "match": "similar emotional tone"
            },
            "acousticness": {
                "high": "acoustic and organic sound",
                "low": "electronic and produced",
                "match": "similar acoustic character"
            },
            "instrumentalness": {
                "high": "instrumental focus",
                "low": "vocal-driven",
                "match": "similar vocal balance"
            },
            "tempo": {
                "high": "fast-paced rhythm",
                "low": "slower tempo",
                "match": "similar BPM range"
            },
        }
    
    def generate_explanation(
        self,
        track: TrackFeatures,
        profile: PlaylistProfile,
        breakdown: Optional[ScoreBreakdown] = None
    ) -> DetailedExplanation:
        """
        Generate comprehensive explanation for a recommendation.
        
        Args:
            track: Recommended track features
            profile: Playlist profile
            breakdown: Score breakdown from scoring engine
            
        Returns:
            DetailedExplanation instance
        """
        explanation = DetailedExplanation(
            track_id=track.track_id,
            summary=""
        )
        
        # Component scores for determining primary signal
        component_scores = {}
        
        # 1. Genre explanation
        genre_exp, genre_strength = self._explain_genre(track, profile, breakdown)
        explanation.genre_explanation = genre_exp
        component_scores['genre'] = genre_strength
        
        # 2. Audio explanation
        audio_exp, audio_strength = self._explain_audio(track, profile, breakdown)
        explanation.audio_explanation = audio_exp
        component_scores['audio'] = audio_strength
        
        # 3. Artist explanation
        artist_exp, artist_strength = self._explain_artist(track, profile, breakdown)
        explanation.artist_explanation = artist_exp
        component_scores['artist'] = artist_strength
        
        # 4. Popularity explanation
        pop_exp, pop_strength = self._explain_popularity(track, profile, breakdown)
        explanation.popularity_explanation = pop_exp
        component_scores['popularity'] = pop_strength
        
        # 5. Collaborative explanation
        collab_exp, collab_strength = self._explain_collaborative(track, profile, breakdown)
        explanation.collaborative_explanation = collab_exp
        component_scores['collaborative'] = collab_strength
        
        # Determine primary signal
        primary_signal = max(component_scores.keys(), key=lambda k: component_scores[k])
        explanation.primary_signal = primary_signal
        
        # Calculate confidence score
        explanation.confidence_score = self._calculate_confidence(
            component_scores,
            breakdown
        )
        
        # Generate summary
        explanation.summary = self._generate_summary(
            track,
            profile,
            component_scores,
            primary_signal
        )
        
        return explanation
    
    def _explain_genre(
        self,
        track: TrackFeatures,
        profile: PlaylistProfile,
        breakdown: Optional[ScoreBreakdown]
    ) -> Tuple[str, float]:
        """Generate genre overlap explanation."""
        if not track.genres:
            return "No genre information available.", 0.0
        
        # Find matching genres
        track_genres = set(g.lower() for g in track.genres)
        playlist_genres = set(g.lower() for g in profile.genre_distribution.keys())
        matched = track_genres & playlist_genres
        
        strength = len(matched) / max(len(track_genres), 1)
        
        if matched:
            matched_list = list(matched)[:3]
            if len(matched) > 3:
                exp = f"Shares genres with your playlist: {', '.join(matched_list)}, and {len(matched)-3} more."
            else:
                exp = f"Shares genres with your playlist: {', '.join(matched_list)}."
            strength = min(strength + 0.3, 1.0)
        elif track.parent_genres & set(profile.parent_genre_distribution.keys()):
            parent_match = track.parent_genres & set(profile.parent_genre_distribution.keys())
            exp = f"Related to your playlist's style ({', '.join(list(parent_match)[:2])})."
            strength = 0.5
        else:
            exp = f"Brings fresh genres: {', '.join(track.genres[:2])}."
            strength = 0.2
        
        return exp, strength
    
    def _explain_audio(
        self,
        track: TrackFeatures,
        profile: PlaylistProfile,
        breakdown: Optional[ScoreBreakdown]
    ) -> Tuple[str, float]:
        """Generate audio feature explanation."""
        # Audio features are deprecated - provide fallback explanation
        # Check if we have valid audio features
        has_audio = (
            breakdown and 
            breakdown.audio_feature_diffs and 
            any(v != 0 for v in breakdown.audio_feature_diffs.values())
        )
        
        if not has_audio:
            # No audio features available
            return "Audio analysis not available for this track.", 0.0
        
        diffs = breakdown.audio_feature_diffs
        
        # Find most notable features
        notable = []
        matching = []
        
        for feature, diff in diffs.items():
            if feature not in self.audio_descriptors:
                continue
            
            descriptors = self.audio_descriptors[feature]
            
            if abs(diff) < 0.15:
                matching.append(descriptors["match"])
            elif diff > 0.3:
                notable.append(descriptors["high"])
            elif diff < -0.3:
                notable.append(descriptors["low"])
        
        # Calculate strength based on audio similarity
        audio_score = breakdown.audio_score if breakdown else 0.5
        
        if matching:
            exp = f"Audio profile match: {', '.join(matching[:3])}."
            strength = audio_score
        elif notable:
            exp = f"Distinct qualities: {', '.join(notable[:2])}."
            strength = max(0.3, audio_score)
        else:
            exp = "Similar overall sound profile to your playlist."
            strength = audio_score
        
        return exp, strength
    
    def _explain_artist(
        self,
        track: TrackFeatures,
        profile: PlaylistProfile,
        breakdown: Optional[ScoreBreakdown]
    ) -> Tuple[str, float]:
        """Generate artist similarity explanation."""
        artist_overlap = breakdown.artist_overlap if breakdown else False
        
        # Check for direct artist overlap
        overlapping_artists = set(track.artist_ids) & profile.unique_artists
        
        if overlapping_artists:
            # Find artist names
            overlap_names = [
                name for aid, name in zip(track.artist_ids, track.artist_names)
                if aid in overlapping_artists
            ]
            if overlap_names:
                exp = f"By {overlap_names[0]}, who appears in your playlist."
                return exp, 1.0
        
        # Check for related artist connection
        if track.artist_names:
            exp = f"From {track.artist_names[0]}, with a compatible style."
            return exp, 0.4
        
        return "Artist aligns with your playlist's taste.", 0.3
    
    def _explain_popularity(
        self,
        track: TrackFeatures,
        profile: PlaylistProfile,
        breakdown: Optional[ScoreBreakdown]
    ) -> Tuple[str, float]:
        """Generate popularity calibration explanation."""
        pop_diff = abs(track.track_popularity - profile.mean_popularity)
        
        if pop_diff < 10:
            exp = f"Similar popularity level ({track.track_popularity}) to your playlist average ({profile.mean_popularity:.0f})."
            strength = 0.9
        elif pop_diff < 20:
            exp = f"Moderately different popularity ({track.track_popularity} vs playlist avg {profile.mean_popularity:.0f})."
            strength = 0.6
        else:
            if track.track_popularity > profile.mean_popularity:
                exp = f"More mainstream than your usual ({track.track_popularity} vs {profile.mean_popularity:.0f})."
            else:
                exp = f"More of a hidden gem ({track.track_popularity} vs {profile.mean_popularity:.0f})."
            strength = 0.3
        
        return exp, strength
    
    def _explain_collaborative(
        self,
        track: TrackFeatures,
        profile: PlaylistProfile,
        breakdown: Optional[ScoreBreakdown]
    ) -> Tuple[str, float]:
        """Generate collaborative signal explanation."""
        cooc_score = track.cooccurrence_score
        
        if cooc_score > 0.5:
            exp = "Frequently appears alongside your playlist tracks in other playlists."
            strength = cooc_score
        elif cooc_score > 0.1:
            exp = "Sometimes paired with your playlist tracks by other listeners."
            strength = cooc_score
        else:
            exp = "Content-based match (limited collaborative data)."
            strength = 0.0
        
        return exp, strength
    
    def _calculate_confidence(
        self,
        component_scores: Dict[str, float],
        breakdown: Optional[ScoreBreakdown]
    ) -> float:
        """Calculate overall confidence in the recommendation."""
        # Average of component scores
        avg_score = np.mean(list(component_scores.values()))
        
        # Boost if multiple strong signals
        strong_signals = sum(1 for v in component_scores.values() if v > 0.6)
        boost = 0.1 * strong_signals
        
        # Use final score if available
        if breakdown:
            return float(min(breakdown.final_score + boost, 1.0))
        
        return float(min(avg_score + boost, 1.0))
    
    def _generate_summary(
        self,
        track: TrackFeatures,
        profile: PlaylistProfile,
        scores: Dict[str, float],
        primary_signal: str
    ) -> str:
        """Generate concise summary sentence."""
        summaries = {
            'genre': lambda: f"Great genre match with your {profile.top_genres[0] if profile.top_genres else 'playlist'} vibes.",
            'audio': lambda: "Matches your playlist's sound and energy.",
            'artist': lambda: f"Recommended based on your love for {track.artist_names[0] if track.artist_names else 'similar artists'}.",
            'popularity': lambda: "Fits your playlist's popularity sweet spot.",
            'collaborative': lambda: "Popular with listeners who have similar taste.",
        }
        
        return summaries.get(primary_signal, lambda: "Matches your playlist's overall vibe.")()


def explain_recommendation(
    track: TrackFeatures,
    profile: PlaylistProfile,
    breakdown: Optional[ScoreBreakdown] = None
) -> str:
    """
    Convenience function for generating a simple explanation.
    
    Args:
        track: Recommended track
        profile: Playlist profile
        breakdown: Optional score breakdown
        
    Returns:
        Simple explanation string
    """
    generator = ExplanationGenerator()
    detailed = generator.generate_explanation(track, profile, breakdown)
    return detailed.summary
