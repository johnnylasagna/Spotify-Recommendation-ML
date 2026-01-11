"""
Command-Line Interface for VibeCheck Recs
==========================================

Usage:
    python -m vibecheck_recs.cli <playlist_url> [options]
    
    or
    
    python cli.py <playlist_url> [options]

Options:
    --num, -n       Number of recommendations (default: 10)
    --cold-start    Simulate cold-start by hiding 50% of tracks
    --output, -o    Output file path (default: stdout)
    --format        Output format: json or csv (default: json)
    --verbose, -v   Verbose output with progress details
    --help, -h      Show this help message

Examples:
    python -m vibecheck_recs.cli https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M
    python -m vibecheck_recs.cli spotify:playlist:37i9dQZF1DXcBWIGoYBM5M -n 5 --cold-start
    python -m vibecheck_recs.cli <playlist_id> -o recommendations.json
"""

import argparse
import json
import sys
import os
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibecheck_recs.recommender import RecommendationEngine, RecommendationOutput
from vibecheck_recs.config import NUM_RECOMMENDATIONS


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog='vibecheck_recs',
        description='🎵 VibeCheck Recs - Spotify-style Playlist Recommendations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://open.spotify.com/playlist/xxxxx
  %(prog)s spotify:playlist:xxxxx -n 5
  %(prog)s <playlist_id> --cold-start -o recs.json

Environment Variables:
  SPOTIFY_CLIENT_ID      Your Spotify API client ID
  SPOTIFY_CLIENT_SECRET  Your Spotify API client secret
        """
    )
    
    parser.add_argument(
        'playlist',
        type=str,
        help='Spotify playlist URL, URI, or ID'
    )
    
    parser.add_argument(
        '-n', '--num',
        type=int,
        default=NUM_RECOMMENDATIONS,
        help=f'Number of recommendations to generate (default: {NUM_RECOMMENDATIONS})'
    )
    
    parser.add_argument(
        '--cold-start',
        action='store_true',
        help='Simulate cold-start scenario (hide 50%% of playlist tracks)'
    )
    
    parser.add_argument(
        '--cold-start-ratio',
        type=float,
        default=0.5,
        help='Ratio of tracks to hide in cold-start mode (default: 0.5)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path (default: print to stdout)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'simple'],
        default='json',
        help='Output format (default: json)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable API response caching'
    )
    
    return parser


def format_output(result: RecommendationOutput, fmt: str) -> str:
    """Format recommendation output based on requested format."""
    if fmt == 'json':
        return result.to_json(indent=2)
    
    elif fmt == 'csv':
        lines = ['track_id,track_name,artists,score,explanation']
        for rec in result.recommendations:
            artists = ';'.join(rec.artist_names)
            # Escape quotes in explanation
            explanation = rec.explanation.replace('"', '""')
            lines.append(
                f'{rec.track_id},"{rec.track_name}","{artists}",{rec.score:.4f},"{explanation}"'
            )
        return '\n'.join(lines)
    
    elif fmt == 'simple':
        lines = [
            f"🎵 Recommendations for: {result.playlist_name}",
            f"   Playlist ID: {result.playlist_id}",
            f"   Track count: {result.track_count}",
            "",
            "Top {0} Recommendations:".format(len(result.recommendations)),
            "-" * 50,
        ]
        for i, rec in enumerate(result.recommendations, 1):
            lines.append(f"{i:2}. {rec.track_name}")
            lines.append(f"    Artists: {', '.join(rec.artist_names)}")
            lines.append(f"    Score: {rec.score:.4f}")
            lines.append(f"    Why: {rec.explanation}")
            lines.append(f"    Track ID: {rec.track_id}")
            lines.append("")
        return '\n'.join(lines)
    
    return result.to_json()


def validate_environment() -> bool:
    """Check if required environment variables are set."""
    client_id = os.environ.get('SPOTIFY_CLIENT_ID')
    client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("❌ Error: Spotify API credentials not found!", file=sys.stderr)
        print("", file=sys.stderr)
        print("Please set the following environment variables:", file=sys.stderr)
        print("  SPOTIFY_CLIENT_ID=your_client_id", file=sys.stderr)
        print("  SPOTIFY_CLIENT_SECRET=your_client_secret", file=sys.stderr)
        print("", file=sys.stderr)
        print("Get credentials at: https://developer.spotify.com/dashboard", file=sys.stderr)
        return False
    
    return True


def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Configure based on verbose flag
    if not args.verbose:
        # Suppress progress output if not verbose
        import io
        sys.stdout = io.StringIO() if args.output else sys.stdout
    
    try:
        # Import SpotifyClient here to handle import errors gracefully
        from vibecheck_recs.spotify_client import SpotifyClient
        
        # Initialize client
        spotify = SpotifyClient(use_cache=not args.no_cache)
        
        # Initialize engine
        engine = RecommendationEngine(spotify_client=spotify)
        
        # Generate recommendations
        result = engine.recommend(
            playlist_input=args.playlist,
            n_recommendations=args.num,
            simulate_cold_start=args.cold_start,
            cold_start_ratio=args.cold_start_ratio
        )
        
        # Reset stdout if we suppressed it
        if not args.verbose and not args.output:
            sys.stdout = sys.__stdout__
        
        # Format output
        output = format_output(result, args.format)
        
        # Write output
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"✅ Recommendations saved to: {args.output}")
        else:
            print(output)
        
        return 0
        
    except Exception as e:
        # Reset stdout on error
        sys.stdout = sys.__stdout__
        
        print(f"❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
