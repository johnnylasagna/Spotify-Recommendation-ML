"""
Candidate Generation Module
============================

Generates candidate tracks for recommendation by exploring:
1. Related artists of playlist artists
2. Albums from playlist artists
3. Genre-based search
4. Playlist co-occurrence (if available)

The goal is to build a diverse yet relevant candidate pool
that likely contains tracks similar to what Spotify would recommend.
"""

import random
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter, defaultdict

import numpy as np

from .spotify_client import SpotifyClient
from .features import (
    FeatureExtractor,
    TrackFeatures,
    PlaylistProfile,
    CollaborativeFeatures,
)
from .config import (
    DEFAULT_CANDIDATE_CONFIG,
    CandidateConfig,
)


class CandidateGenerator:
    """
    Generates candidate tracks for recommendation.
    
    Strategy:
        1. Explore related artists of playlist artists
        2. Fetch top tracks and album tracks from related artists
        3. Search by dominant genres
        4. Filter out existing playlist tracks
        5. Rank by initial relevance signals
    """
    
    def __init__(
        self,
        spotify_client: SpotifyClient,
        feature_extractor: FeatureExtractor,
        config: CandidateConfig = DEFAULT_CANDIDATE_CONFIG
    ):
        """
        Initialize candidate generator.
        
        Args:
            spotify_client: Spotify API client
            feature_extractor: Feature extraction module
            config: Candidate generation configuration
        """
        self.spotify = spotify_client
        self.extractor = feature_extractor
        self.config = config
        
        # Collaborative features (can be populated from external data)
        self.collab_features = CollaborativeFeatures()
    
    def generate_candidates(
        self,
        profile: PlaylistProfile,
        exclude_track_ids: Optional[Set[str]] = None
    ) -> Tuple[List[Dict], List[TrackFeatures]]:
        """
        Generate candidate tracks based on playlist profile.
        
        Args:
            profile: Playlist profile with aggregated features
            exclude_track_ids: Track IDs to exclude (already in playlist)
            
        Returns:
            Tuple of (raw track data, extracted features)
        """
        exclude_ids = exclude_track_ids or set(profile.track_ids)
        
        # Collect candidate tracks from multiple sources
        candidates: Dict[str, Dict] = {}  # track_id -> track data
        
        # 1. Artist albums exploration (most reliable - always works)
        print("  → Exploring artist albums...")
        album_candidates = self._explore_artist_albums(profile, exclude_ids)
        candidates.update(album_candidates)
        
        # 2. Genre-based search (reliable fallback)
        print("  → Searching by genre...")
        genre_candidates = self._search_by_genres(profile, exclude_ids)
        candidates.update(genre_candidates)
        
        # 3. Artist name search (additional candidates)
        print("  → Searching by artist names...")
        artist_candidates = self._search_by_artists(profile, exclude_ids)
        candidates.update(artist_candidates)
        
        # 4. Try related artists (may fail with 404 on newer API apps)
        if len(candidates) < 200:
            print("  → Exploring related artists...")
            related_candidates = self._explore_related_artists(profile, exclude_ids)
            candidates.update(related_candidates)
        
        # Remove excluded tracks
        for track_id in exclude_ids:
            candidates.pop(track_id, None)
        
        print(f"  → Found {len(candidates)} unique candidates")
        
        # Limit candidates
        if len(candidates) > self.config.max_candidates:
            # Prioritize by initial relevance (artist overlap, popularity)
            candidates = self._prioritize_candidates(candidates, profile)
        
        # Extract features for all candidates
        candidate_list = list(candidates.values())
        candidate_features = self._extract_candidate_features(
            candidate_list, 
            profile
        )
        
        return candidate_list, candidate_features
    
    def _explore_related_artists(
        self,
        profile: PlaylistProfile,
        exclude_ids: Set[str]
    ) -> Dict[str, Dict]:
        """
        Explore related artists and their tracks.
        
        Args:
            profile: Playlist profile
            exclude_ids: Track IDs to exclude
            
        Returns:
            Dictionary of track_id -> track data
        """
        candidates = {}
        
        # Get unique artists from playlist, weighted by frequency
        artist_ids = list(profile.artist_frequency.keys())
        
        # Sort by frequency to prioritize more common artists
        artist_ids.sort(
            key=lambda x: profile.artist_frequency.get(x, 0), 
            reverse=True
        )
        
        # Limit number of seed artists to explore
        seed_artists = artist_ids[:min(len(artist_ids), 15)]
        
        explored_artists = set()
        
        for artist_id in seed_artists:
            if artist_id in explored_artists:
                continue
            
            # Get related artists
            related = self.spotify.get_related_artists(artist_id)
            
            # Take top N related artists
            for rel_artist in related[:self.config.related_artists_per_seed]:
                rel_id = rel_artist.get('id')
                if not rel_id or rel_id in explored_artists:
                    continue
                
                explored_artists.add(rel_id)
                
                # Get top tracks from related artist
                top_tracks = self.spotify.get_artist_top_tracks(rel_id)
                
                for track in top_tracks[:self.config.top_tracks_per_artist]:
                    track_id = track.get('id')
                    if track_id and track_id not in exclude_ids:
                        candidates[track_id] = track
        
        return candidates
    
    def _explore_artist_albums(
        self,
        profile: PlaylistProfile,
        exclude_ids: Set[str]
    ) -> Dict[str, Dict]:
        """
        Explore albums from playlist artists.
        
        Args:
            profile: Playlist profile
            exclude_ids: Track IDs to exclude
            
        Returns:
            Dictionary of track_id -> track data
        """
        candidates = {}
        
        # Focus on most frequent artists
        top_artists = sorted(
            profile.artist_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for artist_id, freq in top_artists:
            # Get albums
            albums = self.spotify.get_artist_albums(
                artist_id, 
                limit=self.config.albums_per_artist
            )
            
            for album in albums:
                album_id = album.get('id')
                if not album_id:
                    continue
                
                # Get album tracks
                album_tracks = self.spotify.get_album_tracks(album_id)
                
                for track in album_tracks:
                    track_id = track.get('id')
                    if track_id and track_id not in exclude_ids:
                        # Album tracks don't have full info, need to enrich later
                        candidates[track_id] = {
                            'id': track_id,
                            'name': track.get('name'),
                            'artists': track.get('artists', []),
                            'duration_ms': track.get('duration_ms', 0),
                            'explicit': track.get('explicit', False),
                            '_needs_enrichment': True,
                        }
        
        return candidates
    
    def _search_by_artists(
        self,
        profile: PlaylistProfile,
        exclude_ids: Set[str]
    ) -> Dict[str, Dict]:
        """
        Search for tracks by artist names from the playlist.
        
        Args:
            profile: Playlist profile
            exclude_ids: Track IDs to exclude
            
        Returns:
            Dictionary of track_id -> track data
        """
        candidates = {}
        
        # Get top artists by frequency
        top_artist_ids = sorted(
            profile.artist_frequency.keys(),
            key=lambda x: profile.artist_frequency.get(x, 0),
            reverse=True
        )[:10]
        
        # Get artist names from track features
        artist_names = set()
        for tf in profile.track_features:
            for name in tf.artist_names:
                if name:
                    artist_names.add(name)
        
        # Search for each top artist
        for artist_name in list(artist_names)[:8]:
            tracks = self.spotify.search_tracks(f'artist:"{artist_name}"', limit=30)
            
            for track in tracks:
                track_id = track.get('id')
                if track_id and track_id not in exclude_ids:
                    candidates[track_id] = track
        
        return candidates
    
    def _search_by_genres(
        self,
        profile: PlaylistProfile,
        exclude_ids: Set[str]
    ) -> Dict[str, Dict]:
        """
        Search for tracks by playlist's dominant genres.
        
        Args:
            profile: Playlist profile
            exclude_ids: Track IDs to exclude
            
        Returns:
            Dictionary of track_id -> track data
        """
        candidates = {}
        
        # Use top genres from profile - get more genres
        top_genres = profile.top_genres[:8] if profile.top_genres else []
        
        for genre in top_genres:
            tracks = self.spotify.search_tracks_by_genre(genre, limit=self.config.genre_search_limit)
            
            for track in tracks:
                track_id = track.get('id')
                if track_id and track_id not in exclude_ids:
                    candidates[track_id] = track
        
        # Also search by parent genres
        for parent_genre in list(profile.parent_genre_distribution.keys())[:5]:
            tracks = self.spotify.search_tracks_by_genre(parent_genre, limit=self.config.genre_search_limit)
            
            for track in tracks:
                track_id = track.get('id')
                if track_id and track_id not in exclude_ids:
                    candidates[track_id] = track
        
        return candidates
    
    def _prioritize_candidates(
        self,
        candidates: Dict[str, Dict],
        profile: PlaylistProfile
    ) -> Dict[str, Dict]:
        """
        Prioritize candidates when we have too many.
        
        Uses quick heuristics:
        - Artist overlap with playlist (HEAVILY weighted)
        - Genre overlap signals
        - Popularity similarity
        - Random sampling to maintain diversity
        
        Args:
            candidates: All candidate tracks
            profile: Playlist profile
            
        Returns:
            Prioritized subset of candidates
        """
        scored_candidates = []
        playlist_genres = profile.get_all_genres()
        playlist_genres_lower = set(g.lower() for g in playlist_genres)
        
        for track_id, track in candidates.items():
            score = 0.0
            
            # Artist overlap bonus (VERY HIGH - same artist tracks are gold)
            artist_match = False
            for artist in track.get('artists', []):
                if artist.get('id') in profile.unique_artists:
                    artist_match = True
                    # Weight by how often this artist appears
                    freq = profile.artist_frequency.get(artist.get('id'), 0)
                    score += 5.0 + (freq * 0.5)  # Strong base + frequency bonus
            
            # Genre overlap bonus (check artist genres if available)
            # This is a rough heuristic since we don't have full track data yet
            
            # Popularity similarity
            track_pop = track.get('popularity', 50)
            pop_diff = abs(track_pop - profile.mean_popularity)
            score += (100 - pop_diff) / 100.0 * 0.5
            
            # Small random factor for diversity (reduced to not override artist match)
            score += random.random() * 0.05
            
            scored_candidates.append((track_id, track, score, artist_match))
        
        # Sort by score (artist matches will be at top due to high weight)
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Take top candidates, ensuring good mix of same-artist and discovery
        result = {}
        artist_match_count = 0
        other_count = 0
        max_artist_matches = int(self.config.max_candidates * 0.6)  # 60% can be same-artist
        max_others = int(self.config.max_candidates * 0.4)  # 40% discovery
        
        for track_id, track, score, is_artist_match in scored_candidates:
            if len(result) >= self.config.max_candidates:
                break
            
            if is_artist_match and artist_match_count < max_artist_matches:
                result[track_id] = track
                artist_match_count += 1
            elif not is_artist_match and other_count < max_others:
                result[track_id] = track
                other_count += 1
            elif is_artist_match:
                # If we've hit other limit but still have artist matches, take them
                result[track_id] = track
                artist_match_count += 1
        
        return result
    
    def _extract_candidate_features(
        self,
        candidates: List[Dict],
        profile: PlaylistProfile
    ) -> List[TrackFeatures]:
        """
        Extract features for all candidate tracks.
        
        Args:
            candidates: List of candidate track data
            profile: Playlist profile (for context)
            
        Returns:
            List of TrackFeatures for each candidate
        """
        if not candidates:
            return []
        
        # Collect track IDs (filter out those needing enrichment)
        track_ids = []
        for c in candidates:
            if c.get('id'):
                track_ids.append(c['id'])
        
        # Fetch full track data, audio features, and artists
        print("  → Fetching track details...")
        tracks, audio_features, artists_dict = self.spotify.get_full_track_data(
            track_ids
        )
        
        # Create mapping for quick lookup
        track_map = {t['id']: t for t in tracks if t}
        audio_map = {}
        for i, tid in enumerate(track_ids):
            if i < len(audio_features):
                audio_map[tid] = audio_features[i]
        
        # Build genre vocabulary from playlist + candidates
        all_genres = []
        for tf in profile.track_features:
            all_genres.append(tf.genres)
        for artist in artists_dict.values():
            all_genres.append(artist.get('genres', []))
        
        self.extractor.build_genre_vocabulary(all_genres)
        
        # Extract features for each candidate
        features = []
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
            
            # Add collaborative features if available
            tf.cooccurrence_score = self.collab_features.get_cooccurrence_score(
                track_id,
                set(profile.track_ids)
            )
            
            features.append(tf)
        
        return features


class DiversityFilter:
    """
    Ensures diversity in final recommendations.
    
    Prevents:
    - Too many tracks from same artist (but allows more than before)
    - Too many tracks from same album
    - Too similar genre profiles
    """
    
    def __init__(
        self,
        max_per_artist: int = 3,  # Increased from 2
        max_per_album: int = 2,   # Increased from 1
        min_genre_diversity: float = 0.3
    ):
        self.max_per_artist = max_per_artist
        self.max_per_album = max_per_album
        self.min_genre_diversity = min_genre_diversity
    
    def filter(
        self,
        ranked_tracks: List[Tuple[TrackFeatures, float]],
        n_select: int = 10
    ) -> List[Tuple[TrackFeatures, float]]:
        """
        Apply diversity filtering to ranked tracks.
        
        Args:
            ranked_tracks: List of (TrackFeatures, score) tuples, sorted by score
            n_select: Number of tracks to select
            
        Returns:
            Filtered list maintaining diversity
        """
        selected = []
        artist_counts: Dict[str, int] = defaultdict(int)
        album_counts: Dict[str, int] = defaultdict(int)
        selected_genres: List[Set[str]] = []
        
        for tf, score in ranked_tracks:
            if len(selected) >= n_select:
                break
            
            # Check artist limit
            primary_artist = tf.artist_ids[0] if tf.artist_ids else None
            if primary_artist:
                if artist_counts[primary_artist] >= self.max_per_artist:
                    continue
            
            # Check album limit
            if tf.album_id:
                if album_counts[tf.album_id] >= self.max_per_album:
                    continue
            
            # Check genre diversity (avoid too similar genre profiles)
            track_genres = set(g.lower() for g in tf.genres)
            if selected_genres and track_genres:
                too_similar = False
                for existing_genres in selected_genres:
                    if existing_genres:
                        overlap = len(track_genres & existing_genres)
                        union = len(track_genres | existing_genres)
                        if union > 0 and overlap / union > 0.8:  # 80% overlap = too similar
                            too_similar = True
                            break
                if too_similar:
                    continue
            
            # Add track
            selected.append((tf, score))
            if primary_artist:
                artist_counts[primary_artist] += 1
            if tf.album_id:
                album_counts[tf.album_id] += 1
            selected_genres.append(track_genres)
        
        return selected
