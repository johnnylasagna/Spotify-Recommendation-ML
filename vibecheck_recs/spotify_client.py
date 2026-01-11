"""
Spotify API Client Wrapper
==========================

Handles all interactions with Spotify API including:
- Authentication
- Track/Artist/Album metadata fetching
- Audio features retrieval
- Playlist data extraction
- Rate limiting and caching
"""

import os
import json
import time
import hashlib
from typing import List, Dict, Optional, Set, Tuple, Any
from functools import lru_cache
from pathlib import Path

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

from .config import (
    SPOTIFY_CLIENT_ID,
    SPOTIFY_CLIENT_SECRET, 
    SPOTIFY_REDIRECT_URI,
    CACHE_DIR,
    CACHE_TTL_HOURS,
)


class SpotifyClient:
    """
    Wrapper around Spotipy with caching and batch operations.
    
    Attributes:
        sp: Spotipy client instance
        cache_enabled: Whether to use local caching
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize Spotify client with credentials.
        
        Args:
            use_cache: Enable local caching for API responses
        """
        self.cache_enabled = use_cache
        self.cache_dir = Path(CACHE_DIR)
        
        if use_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        # Get credentials from environment at runtime (not import time)
        client_id = os.environ.get("SPOTIFY_CLIENT_ID") or os.environ.get("SPOTIPY_CLIENT_ID") or SPOTIFY_CLIENT_ID
        client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET") or os.environ.get("SPOTIPY_CLIENT_SECRET") or SPOTIFY_CLIENT_SECRET
        
        # Initialize Spotipy with client credentials flow
        auth_manager = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        self.sp = spotipy.Spotify(auth_manager=auth_manager)
        
        # Request throttling
        self._last_request_time = 0
        self._min_request_interval = 0.05  # 50ms between requests
    
    def _throttle(self):
        """Ensure minimum time between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _get_cache_key(self, prefix: str, data: Any) -> str:
        """Generate cache key from data."""
        data_str = json.dumps(data, sort_keys=True)
        hash_val = hashlib.md5(data_str.encode()).hexdigest()[:12]
        return f"{prefix}_{hash_val}"
    
    def _load_cache(self, key: str) -> Optional[Dict]:
        """Load data from cache if valid."""
        if not self.cache_enabled:
            return None
            
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            
            # Check TTL
            if time.time() - cached.get('timestamp', 0) > CACHE_TTL_HOURS * 3600:
                return None
                
            return cached.get('data')
        except (json.JSONDecodeError, IOError):
            return None
    
    def _save_cache(self, key: str, data: Any):
        """Save data to cache."""
        if not self.cache_enabled:
            return
            
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({'timestamp': time.time(), 'data': data}, f)
        except IOError:
            pass  # Silently fail on cache write errors
    
    # =========================================================================
    # PLAYLIST OPERATIONS
    # =========================================================================
    
    def get_playlist(self, playlist_id: str) -> Dict:
        """
        Fetch playlist metadata and tracks.
        
        Args:
            playlist_id: Spotify playlist ID or URI
            
        Returns:
            Playlist data including tracks
        """
        cache_key = self._get_cache_key("playlist", playlist_id)
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        self._throttle()
        
        # Extract ID from URL/URI if needed
        if "playlist/" in playlist_id:
            playlist_id = playlist_id.split("playlist/")[-1].split("?")[0]
        
        playlist = self.sp.playlist(playlist_id)
        
        # Fetch all tracks (handle pagination)
        tracks = []
        results = playlist['tracks']
        while results:
            for item in results['items']:
                if item['track'] and item['track']['id']:
                    tracks.append(item['track'])
            
            if results['next']:
                self._throttle()
                results = self.sp.next(results)
            else:
                results = None
        
        playlist['all_tracks'] = tracks
        self._save_cache(cache_key, playlist)
        
        return playlist
    
    def extract_playlist_id(self, playlist_input: str) -> str:
        """
        Extract playlist ID from various input formats.
        
        Args:
            playlist_input: Playlist URL, URI, or ID
            
        Returns:
            Clean playlist ID
        """
        if "playlist/" in playlist_input:
            return playlist_input.split("playlist/")[-1].split("?")[0]
        elif "spotify:playlist:" in playlist_input:
            return playlist_input.split("spotify:playlist:")[-1]
        return playlist_input
    
    # =========================================================================
    # TRACK OPERATIONS
    # =========================================================================
    
    def get_tracks(self, track_ids: List[str]) -> List[Dict]:
        """
        Fetch track metadata in batches.
        
        Args:
            track_ids: List of Spotify track IDs
            
        Returns:
            List of track metadata dictionaries
        """
        if not track_ids:
            return []
        
        cache_key = self._get_cache_key("tracks", sorted(track_ids))
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        tracks = []
        # Spotify API limit: 50 tracks per request
        for i in range(0, len(track_ids), 50):
            batch = track_ids[i:i+50]
            self._throttle()
            try:
                result = self.sp.tracks(batch)
                tracks.extend([t for t in result['tracks'] if t])
            except Exception as e:
                print(f"Warning: Error fetching tracks batch: {e}")
        
        self._save_cache(cache_key, tracks)
        return tracks
    
    def get_audio_features(self, track_ids: List[str]) -> List[Optional[Dict]]:
        """
        Fetch audio features for tracks in batches.
        
        Args:
            track_ids: List of Spotify track IDs
            
        Returns:
            List of audio feature dictionaries (None for unavailable)
        """
        if not track_ids:
            return []
        
        cache_key = self._get_cache_key("audio_features", sorted(track_ids))
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        features = []
        # Spotify API limit: 100 tracks per request
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i+100]
            self._throttle()
            try:
                result = self.sp.audio_features(batch)
                features.extend(result if result else [None] * len(batch))
            except Exception as e:
                print(f"Warning: Error fetching audio features: {e}")
                features.extend([None] * len(batch))
        
        self._save_cache(cache_key, features)
        return features
    
    # =========================================================================
    # ARTIST OPERATIONS
    # =========================================================================
    
    def get_artists(self, artist_ids: List[str]) -> List[Dict]:
        """
        Fetch artist metadata in batches.
        
        Args:
            artist_ids: List of Spotify artist IDs
            
        Returns:
            List of artist metadata dictionaries
        """
        if not artist_ids:
            return []
        
        # Deduplicate
        artist_ids = list(set(artist_ids))
        
        cache_key = self._get_cache_key("artists", sorted(artist_ids))
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        artists = []
        # Spotify API limit: 50 artists per request
        for i in range(0, len(artist_ids), 50):
            batch = artist_ids[i:i+50]
            self._throttle()
            try:
                result = self.sp.artists(batch)
                artists.extend([a for a in result['artists'] if a])
            except Exception as e:
                print(f"Warning: Error fetching artists: {e}")
        
        self._save_cache(cache_key, artists)
        return artists
    
    def get_related_artists(self, artist_id: str) -> List[Dict]:
        """
        Fetch related artists for a given artist.
        
        Args:
            artist_id: Spotify artist ID
            
        Returns:
            List of related artist dictionaries
        """
        cache_key = self._get_cache_key("related_artists", artist_id)
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        self._throttle()
        try:
            result = self.sp.artist_related_artists(artist_id)
            artists = result.get('artists', [])
            self._save_cache(cache_key, artists)
            return artists
        except Exception as e:
            print(f"Warning: Error fetching related artists: {e}")
            return []
    
    def get_artist_top_tracks(self, artist_id: str, country: str = 'US') -> List[Dict]:
        """
        Fetch top tracks for an artist.
        
        Args:
            artist_id: Spotify artist ID
            country: Market country code
            
        Returns:
            List of top track dictionaries
        """
        cache_key = self._get_cache_key("top_tracks", f"{artist_id}_{country}")
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        self._throttle()
        try:
            result = self.sp.artist_top_tracks(artist_id, country=country)
            tracks = result.get('tracks', [])
            self._save_cache(cache_key, tracks)
            return tracks
        except Exception as e:
            print(f"Warning: Error fetching top tracks: {e}")
            return []
    
    def get_artist_albums(self, artist_id: str, limit: int = 10) -> List[Dict]:
        """
        Fetch albums for an artist.
        
        Args:
            artist_id: Spotify artist ID
            limit: Maximum albums to fetch
            
        Returns:
            List of album dictionaries
        """
        cache_key = self._get_cache_key("albums", f"{artist_id}_{limit}")
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        self._throttle()
        try:
            result = self.sp.artist_albums(
                artist_id, 
                album_type='album,single',
                limit=limit
            )
            albums = result.get('items', [])
            self._save_cache(cache_key, albums)
            return albums
        except Exception as e:
            print(f"Warning: Error fetching albums: {e}")
            return []
    
    def get_album_tracks(self, album_id: str) -> List[Dict]:
        """
        Fetch tracks from an album.
        
        Args:
            album_id: Spotify album ID
            
        Returns:
            List of track dictionaries
        """
        cache_key = self._get_cache_key("album_tracks", album_id)
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        self._throttle()
        try:
            result = self.sp.album_tracks(album_id)
            tracks = result.get('items', [])
            self._save_cache(cache_key, tracks)
            return tracks
        except Exception as e:
            print(f"Warning: Error fetching album tracks: {e}")
            return []
    
    # =========================================================================
    # SEARCH OPERATIONS (for candidate generation)
    # =========================================================================
    
    def search_tracks_by_genre(
        self, 
        genre: str, 
        limit: int = 50
    ) -> List[Dict]:
        """
        Search for tracks by genre.
        
        Args:
            genre: Genre name to search
            limit: Maximum tracks to return
            
        Returns:
            List of track dictionaries
        """
        cache_key = self._get_cache_key("search_genre", f"{genre}_{limit}")
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        self._throttle()
        try:
            result = self.sp.search(
                q=f'genre:"{genre}"',
                type='track',
                limit=min(limit, 50)
            )
            tracks = result.get('tracks', {}).get('items', [])
            self._save_cache(cache_key, tracks)
            return tracks
        except Exception as e:
            print(f"Warning: Error searching by genre: {e}")
            return []
    
    def search_tracks(
        self, 
        query: str, 
        limit: int = 50
    ) -> List[Dict]:
        """
        General track search.
        
        Args:
            query: Search query
            limit: Maximum tracks to return
            
        Returns:
            List of track dictionaries
        """
        cache_key = self._get_cache_key("search", f"{query}_{limit}")
        cached = self._load_cache(cache_key)
        if cached:
            return cached
        
        self._throttle()
        try:
            result = self.sp.search(
                q=query,
                type='track',
                limit=min(limit, 50)
            )
            tracks = result.get('tracks', {}).get('items', [])
            self._save_cache(cache_key, tracks)
            return tracks
        except Exception as e:
            print(f"Warning: Error searching tracks: {e}")
            return []
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_full_track_data(
        self, 
        track_ids: List[str]
    ) -> Tuple[List[Dict], List[Optional[Dict]], Dict[str, Dict]]:
        """
        Fetch complete track data including audio features and artist info.
        
        Args:
            track_ids: List of track IDs
            
        Returns:
            Tuple of (tracks, audio_features, artists_dict)
        """
        # Get tracks
        tracks = self.get_tracks(track_ids)
        
        # Get audio features
        valid_ids = [t['id'] for t in tracks if t]
        audio_features = self.get_audio_features(valid_ids)
        
        # Get unique artist IDs
        artist_ids = set()
        for track in tracks:
            if track:
                for artist in track.get('artists', []):
                    if artist.get('id'):
                        artist_ids.add(artist['id'])
        
        # Get artist data
        artists_list = self.get_artists(list(artist_ids))
        artists_dict = {a['id']: a for a in artists_list if a}
        
        return tracks, audio_features, artists_dict
