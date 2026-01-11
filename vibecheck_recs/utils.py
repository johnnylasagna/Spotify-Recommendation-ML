"""
Utility Functions
=================

Common utilities used across the VibeCheck Recs system.
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Callable
from functools import wraps


class Cache:
    """Simple file-based cache with TTL support."""
    
    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 24):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory for cache files
            ttl_hours: Time-to-live in hours
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_hours * 3600
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_key(self, data: Any) -> str:
        """Generate cache key from data."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            
            # Check TTL
            if time.time() - cached.get('timestamp', 0) > self.ttl_seconds:
                cache_file.unlink()
                return None
            
            return cached.get('data')
        except (json.JSONDecodeError, IOError):
            return None
    
    def set(self, key: str, data: Any) -> None:
        """Set value in cache."""
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': time.time(),
                    'data': data
                }, f)
        except IOError:
            pass
    
    def clear(self) -> int:
        """Clear all cache files. Returns number of files deleted."""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except IOError:
                pass
        return count


def cached(cache: Cache, key_prefix: str = ""):
    """
    Decorator for caching function results.
    
    Args:
        cache: Cache instance
        key_prefix: Prefix for cache keys
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key from function args
            key_data = {
                'prefix': key_prefix or func.__name__,
                'args': args[1:] if args else (),  # Skip self
                'kwargs': kwargs
            }
            key = cache._get_key(key_data)
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.set(key, result)
            
            return result
        return wrapper
    return decorator


class ProgressTracker:
    """Simple progress tracking for long operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            description: Description to display
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """Update progress by n items."""
        self.current += n
        self._display()
    
    def _display(self):
        """Display current progress."""
        pct = (self.current / self.total) * 100 if self.total > 0 else 0
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"
        
        print(f"\r  {self.description}: {self.current}/{self.total} ({pct:.1f}%) | {eta_str}", end="")
        
        if self.current >= self.total:
            print()  # New line when done
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if self.current < self.total:
            print()  # Ensure new line on exit


def normalize_playlist_url(url: str) -> str:
    """
    Normalize various playlist URL formats to playlist ID.
    
    Args:
        url: Spotify playlist URL, URI, or ID
        
    Returns:
        Clean playlist ID
    """
    # Handle full URLs
    if "spotify.com/playlist/" in url:
        # Extract ID from URL
        playlist_id = url.split("/playlist/")[-1].split("?")[0]
    elif "spotify:playlist:" in url:
        # Handle Spotify URI
        playlist_id = url.split("spotify:playlist:")[-1]
    else:
        # Assume it's already an ID
        playlist_id = url.strip()
    
    return playlist_id


def validate_track_id(track_id: str) -> bool:
    """
    Validate Spotify track ID format.
    
    Args:
        track_id: Track ID to validate
        
    Returns:
        True if valid format
    """
    if not track_id:
        return False
    
    # Spotify IDs are 22 characters, base62
    if len(track_id) != 22:
        return False
    
    valid_chars = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    return all(c in valid_chars for c in track_id)


def batch_process(items: list, batch_size: int, processor: Callable) -> list:
    """
    Process items in batches.
    
    Args:
        items: Items to process
        batch_size: Size of each batch
        processor: Function to call on each batch
        
    Returns:
        Flattened list of results
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_result = processor(batch)
        if isinstance(batch_result, list):
            results.extend(batch_result)
        else:
            results.append(batch_result)
    
    return results
