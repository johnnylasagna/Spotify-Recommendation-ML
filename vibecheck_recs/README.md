# 🎵 VibeCheck Recs

**A Spotify-style Recommendation System for the NEXUS ML Hackathon**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

VibeCheck Recs is a custom-built music recommendation engine that analyzes Spotify playlists and suggests tracks that match the playlist's vibe. It uses a hybrid approach combining **content-based filtering** and **collaborative signals** to maximize similarity with Spotify's own recommendations.

### Key Features

- **Hybrid Scoring**: Combines audio features, genre overlap, artist similarity, and popularity calibration
- **Cold-Start Robust**: Adapts weights when only partial playlist data is available
- **Explainable**: Generates human-readable explanations for each recommendation
- **Fast Inference**: Caching and batch operations for efficient API usage
- **No LLMs**: Pure ML/algorithmic approach as per hackathon constraints

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     VIBECHECK RECS PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │ Playlist │───>│   Feature    │───>│  Candidate Generation │  │
│  │   URL    │    │  Extraction  │    │                       │  │
│  └──────────┘    └──────────────┘    └───────────────────────┘  │
│                                                 │               │
│                                                 ▼               │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Output  │<───│   Ranking    │<───│   Hybrid Scoring      │  │
│  │   JSON   │    │ + Diversity  │    │                       │  │
│  └──────────┘    └──────────────┘    └───────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
vibecheck_recs/
├── __init__.py          # Package initialization
├── __main__.py          # Module entry point
├── config.py            # Configuration and constants
├── spotify_client.py    # Spotify API wrapper with caching
├── features.py          # Feature extraction and engineering
├── candidates.py        # Candidate track generation
├── scoring.py           # Hybrid scoring engine
├── explainer.py         # Explanation generation
├── recommender.py       # Main orchestration engine
└── cli.py               # Command-line interface
```

## 🔬 Algorithm

### Scoring Function

The final recommendation score is computed as a weighted combination:

$$
\text{Score} = \sum_{i} w_i \times S_i
$$

Where components are:

| Component | Weight | Formula |
|-----------|--------|---------|
| Audio Similarity | 0.30 | $\cos(\mathbf{v}_{candidate}, \mathbf{v}_{centroid})$ |
| Genre Overlap | 0.20 | Hierarchical Jaccard similarity |
| Artist Similarity | 0.15 | Frequency-weighted overlap |
| Collaborative | 0.20 | Normalized co-occurrence score |
| Popularity Match | 0.10 | $1 - \frac{\|p_c - \mu_p\|}{100}$ |
| Diversity Bonus | 0.05 | Variety contribution |

### Audio Features (9-dimensional vector)

- Danceability, Energy, Loudness (normalized)
- Speechiness, Acousticness, Instrumentalness
- Liveness, Valence, Tempo (normalized)

### Genre Matching

Uses hierarchical genre taxonomy:
- **Exact match**: Direct genre overlap (weight: 0.7)
- **Parent match**: Matching genre categories (weight: 0.3)

### Cold-Start Handling

When playlist has < 10 tracks:
- Increases genre weight (0.30)
- Increases artist weight (0.25)
- Reduces collaborative weight (0.05)
- Uses genre expansion for candidate generation

## 🚀 Quick Start

### Prerequisites

1. Python 3.8+
2. Spotify Developer Account with API credentials

### Installation

```bash
# Clone or download the project
cd vibecheck_recs

# Install dependencies
pip install -r requirements.txt
```

### Set Environment Variables

```bash
# Windows (PowerShell)
$env:SPOTIFY_CLIENT_ID = "your_client_id"
$env:SPOTIFY_CLIENT_SECRET = "your_client_secret"

# Linux/Mac
export SPOTIFY_CLIENT_ID="your_client_id"
export SPOTIFY_CLIENT_SECRET="your_client_secret"
```

### Run

```bash
# Basic usage
python -m vibecheck_recs "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"

# With options
python -m vibecheck_recs <playlist_url> -n 10 --format simple

# Cold-start simulation
python -m vibecheck_recs <playlist_url> --cold-start

# Save to file
python -m vibecheck_recs <playlist_url> -o recommendations.json
```

## 📊 Output Format

```json
{
  "playlist_id": "37i9dQZF1DXcBWIGoYBM5M",
  "playlist_name": "Today's Top Hits",
  "track_count": 50,
  "recommendations": [
    {
      "track_id": "4iJyoBOLtHqaGxP12qzhQI",
      "track_name": "Example Track",
      "artist_names": ["Artist Name"],
      "score": 0.8721,
      "explanation": "Genre match: pop, dance pop; Similar sound profile"
    }
  ]
}
```

## 🧪 Testing

```bash
# Run with a test playlist
python -m vibecheck_recs "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M" -v

# Test cold-start mode
python -m vibecheck_recs <playlist_url> --cold-start --cold-start-ratio 0.5
```

## 📈 Performance Optimizations

1. **API Caching**: Responses cached locally for 24 hours
2. **Batch Operations**: Tracks/artists fetched in batches (50-100 per request)
3. **Request Throttling**: Automatic rate limiting (50ms between requests)
4. **Early Candidate Filtering**: Prioritization before expensive feature extraction
5. **Precomputed Genre Vocabulary**: TF-IDF weights computed once

## 🎯 Evaluation Alignment

This system is designed to maximize the hackathon evaluation metric:

1. **Cosine Similarity**: Uses normalized feature vectors for optimal cosine matching
2. **Bipartite Matching**: Diverse recommendations improve optimal assignment
3. **Average Similarity**: Weighted scoring focuses on Spotify-like relevance

## 🔧 Configuration

Edit `config.py` to tune:

```python
# Scoring weights
DEFAULT_WEIGHTS = ScoringWeights(
    audio_similarity=0.30,
    genre_overlap=0.20,
    artist_similarity=0.15,
    playlist_cooccurrence=0.20,
    popularity_match=0.10,
    diversity_bonus=0.05,
)

# Candidate generation
DEFAULT_CANDIDATE_CONFIG = CandidateConfig(
    related_artists_per_seed=5,
    top_tracks_per_artist=10,
    max_candidates=500,
)
```

## 📝 Dependencies

```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=0.24.0
scipy>=1.7.0
spotipy>=2.19.0
```

## ⚖️ Constraints Compliance

| Constraint | Status |
|------------|--------|
| ❌ No Spotify recommendation endpoint | ✅ Compliant |
| ❌ No LLMs for recommendations | ✅ Compliant |
| ✅ Custom-built logic only | ✅ Compliant |
| ✅ Track/Artist/Genre/Audio features | ✅ Used |

## 📄 License

MIT License - See LICENSE file for details.

---

Built for the **NEXUS Machine Learning Hackathon – Stage 2 (Builder Round)**
