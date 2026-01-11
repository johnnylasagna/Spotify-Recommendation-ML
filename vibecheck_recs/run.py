#!/usr/bin/env python
"""
VibeCheck Recs - Quick Run Script
==================================

Simple script to run the recommendation engine.

Usage:
    python run.py <playlist_url>
    
Example:
    python run.py "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
"""

import sys
import os

# Ensure the parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vibecheck_recs.recommender import RecommendationEngine


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <playlist_url>")
        print()
        print("Example:")
        print('  python run.py "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"')
        sys.exit(1)
    
    playlist_url = sys.argv[1]
    
    # Check credentials
    if not os.environ.get('SPOTIFY_CLIENT_ID') or not os.environ.get('SPOTIFY_CLIENT_SECRET'):
        print("Error: Spotify credentials not set!")
        print()
        print("Set these environment variables:")
        print("  SPOTIFY_CLIENT_ID=your_client_id")
        print("  SPOTIFY_CLIENT_SECRET=your_client_secret")
        sys.exit(1)
    
    # Run recommendation
    engine = RecommendationEngine()
    result = engine.recommend(playlist_url)
    
    # Print results
    print()
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print()
    
    for i, rec in enumerate(result.recommendations, 1):
        print(f"{i:2}. {rec.track_name}")
        print(f"    Artist: {', '.join(rec.artist_names)}")
        print(f"    Score:  {rec.score:.4f}")
        print(f"    Why:    {rec.explanation}")
        print(f"    ID:     {rec.track_id}")
        print()
    
    # Also save JSON output
    output_file = "recommendations.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result.to_json())
    
    print(f"Full output saved to: {output_file}")


if __name__ == '__main__':
    main()
