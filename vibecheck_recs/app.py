"""
VibeCheck Recs - Streamlit Web App
===================================

A beautiful web interface for the VibeCheck recommendation system.

Run with:
    streamlit run vibecheck_recs/app.py
"""

import streamlit as st
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibecheck_recs.recommender import RecommendationEngine, RecommendationOutput
from vibecheck_recs.config import NUM_RECOMMENDATIONS, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET


# =============================================================================
# COMPATIBILITY HELPERS
# =============================================================================
def safe_rerun():
    """Rerun the app - compatible with old and new Streamlit versions."""
    try:
        st.rerun()
    except AttributeError:
        # Older Streamlit versions use experimental_rerun
        st.experimental_rerun()


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="VibeCheck Recs",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .track-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1DB954;
    }
    .track-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #fff;
        margin-bottom: 0.3rem;
    }
    .artist-name {
        font-size: 0.9rem;
        color: #b3b3b3;
        margin-bottom: 0.5rem;
    }
    .score-badge {
        background: #1DB954;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .explanation {
        font-size: 0.85rem;
        color: #a0a0a0;
        margin-top: 0.5rem;
        font-style: italic;
    }
    .stat-box {
        background: #1a1a2e;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1DB954;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #888;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1DB954, #1ed760);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 25px;
        width: 100%;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_credentials() -> bool:
    """Check if Spotify credentials are configured."""
    client_id = os.environ.get("SPOTIFY_CLIENT_ID", "") or st.session_state.get("spotify_client_id", "")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "") or st.session_state.get("spotify_client_secret", "")
    return bool(client_id and client_secret)


def set_credentials(client_id: str, client_secret: str):
    """Set Spotify credentials in environment."""
    os.environ["SPOTIFY_CLIENT_ID"] = client_id
    os.environ["SPOTIFY_CLIENT_SECRET"] = client_secret
    os.environ["SPOTIPY_CLIENT_ID"] = client_id
    os.environ["SPOTIPY_CLIENT_SECRET"] = client_secret
    st.session_state["spotify_client_id"] = client_id
    st.session_state["spotify_client_secret"] = client_secret
    # Clear cached engine so it picks up new credentials
    get_recommendation_engine.clear()


def extract_playlist_id_from_url(url: str) -> str:
    """Extract playlist ID from various URL formats."""
    url = url.strip()
    
    # Handle full URL
    if "open.spotify.com/playlist/" in url:
        playlist_id = url.split("playlist/")[1].split("?")[0]
        return playlist_id
    
    # Handle URI format
    if url.startswith("spotify:playlist:"):
        return url.replace("spotify:playlist:", "")
    
    # Assume it's already an ID
    return url


@st.cache_resource
def get_recommendation_engine():
    """Get or create the recommendation engine (cached)."""
    # Ensure environment variables are set from session state
    if "spotify_client_id" in st.session_state:
        os.environ["SPOTIFY_CLIENT_ID"] = st.session_state["spotify_client_id"]
        os.environ["SPOTIPY_CLIENT_ID"] = st.session_state["spotify_client_id"]
    if "spotify_client_secret" in st.session_state:
        os.environ["SPOTIFY_CLIENT_SECRET"] = st.session_state["spotify_client_secret"]
        os.environ["SPOTIPY_CLIENT_SECRET"] = st.session_state["spotify_client_secret"]
    return RecommendationEngine()


def render_track_card(rec: dict, index: int):
    """Render a single track recommendation as a card."""
    with st.container():
        col1, col2 = st.columns([5, 1])
        
        with col1:
            st.markdown(f"""
            <div class="track-card">
                <div class="track-name">
                    {index}. {rec['track_name']}
                </div>
                <div class="artist-name">
                    {', '.join(rec['artist_names'])}
                </div>
                <div class="explanation">
                    💡 {rec['explanation']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            score_pct = rec['score'] * 100
            st.metric("Score", f"{score_pct:.1f}%")


def render_recommendations(output: RecommendationOutput):
    """Render the full recommendation output."""
    
    # Header stats
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{output.track_count}</div>
            <div class="stat-label">Playlist Tracks</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{len(output.recommendations)}</div>
            <div class="stat-label">Recommendations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_score = sum(r.score for r in output.recommendations) / len(output.recommendations) if output.recommendations else 0
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-value">{avg_score*100:.1f}%</div>
            <div class="stat-label">Avg. Match Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommendations list
    st.subheader("🎶 Recommended Tracks")
    
    for i, rec in enumerate(output.recommendations, 1):
        rec_dict = {
            "track_id": rec.track_id,
            "track_name": rec.track_name,
            "artist_names": rec.artist_names,
            "score": rec.score,
            "explanation": rec.explanation
        }
        render_track_card(rec_dict, i)
    
    # Export options
    st.markdown("---")
    st.subheader("📥 Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        json_data = output.to_json()
        st.download_button(
            label="📄 Download JSON",
            data=json_data,
            file_name=f"vibecheck_recs_{output.playlist_id}.json",
            mime="application/json"
        )
    
    with col2:
        # Create track ID list for easy copying
        track_ids = [r.track_id for r in output.recommendations]
        track_ids_str = "\n".join(track_ids)
        st.download_button(
            label="🎵 Download Track IDs",
            data=track_ids_str,
            file_name=f"track_ids_{output.playlist_id}.txt",
            mime="text/plain"
        )


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🎵 VibeCheck Recs</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Spotify Playlist Recommendations</p>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Credentials section
        with st.expander("🔑 Spotify Credentials", expanded=not check_credentials()):
            st.markdown("""
            Enter your Spotify API credentials. Get them from the 
            [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).
            """)
            
            client_id = st.text_input(
                "Client ID",
                value=st.session_state.get("spotify_client_id", ""),
                type="password"
            )
            client_secret = st.text_input(
                "Client Secret",
                value=st.session_state.get("spotify_client_secret", ""),
                type="password"
            )
            
            if st.button("Save Credentials"):
                if client_id and client_secret:
                    set_credentials(client_id, client_secret)
                    st.success("✅ Credentials saved!")
                    safe_rerun()
                else:
                    st.error("Please enter both Client ID and Secret")
        
        st.markdown("---")
        
        # Recommendation settings
        st.subheader("🎛️ Recommendation Settings")
        
        num_recs = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
        
        cold_start = st.checkbox(
            "🥶 Simulate Cold Start",
            value=False,
            help="Hide 50% of playlist tracks to test cold-start robustness"
        )
        
        if cold_start:
            cold_start_ratio = st.slider(
                "Hidden track ratio",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1
            )
        else:
            cold_start_ratio = 0.5
        
        st.markdown("---")
        
        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **VibeCheck Recs** is an AI-powered recommendation system 
            that analyzes your Spotify playlists and suggests similar tracks.
            
            **How it works:**
            1. 📊 Analyzes your playlist's genres, artists, and style
            2. 🔍 Discovers candidate tracks from similar artists/albums
            3. 📈 Scores candidates using a hybrid algorithm
            4. 🎯 Returns diverse, high-quality recommendations
            
            Built for the NEXUS ML Hackathon.
            """)
    
    # Main content area
    if not check_credentials():
        st.warning("⚠️ Please enter your Spotify API credentials in the sidebar to get started.")
        
        st.markdown("""
        ### Getting Started
        
        1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
        2. Create a new app (or use an existing one)
        3. Copy your **Client ID** and **Client Secret**
        4. Paste them in the sidebar
        """)
        return
    
    # Playlist input
    st.markdown("### 🎧 Enter a Spotify Playlist")
    
    playlist_url = st.text_input(
        "Playlist URL or ID",
        placeholder="https://open.spotify.com/playlist/xxxxx or spotify:playlist:xxxxx",
        help="Paste a Spotify playlist URL, URI, or just the playlist ID"
    )
    
    # Generate button
    if st.button("🚀 Generate Recommendations", type="primary", use_container_width=True):
        if not playlist_url:
            st.error("Please enter a playlist URL")
            return
        
        try:
            # Progress indicators
            progress_container = st.empty()
            status_container = st.empty()
            
            with st.spinner(""):
                # Create progress bar
                progress_bar = progress_container.progress(0)
                
                # Step 1: Initialize
                status_container.info("🔄 Initializing recommendation engine...")
                progress_bar.progress(10)
                
                engine = get_recommendation_engine()
                
                # Step 2: Fetch playlist
                status_container.info("📥 Fetching playlist data...")
                progress_bar.progress(20)
                
                # Step 3: Extract features
                status_container.info("🔬 Analyzing playlist features...")
                progress_bar.progress(40)
                
                # Step 4: Generate candidates
                status_container.info("🔍 Discovering candidate tracks...")
                progress_bar.progress(60)
                
                # Run the recommendation
                result = engine.recommend(
                    playlist_input=playlist_url,
                    n_recommendations=num_recs,
                    simulate_cold_start=cold_start,
                    cold_start_ratio=cold_start_ratio
                )
                
                # Step 5: Scoring
                status_container.info("📊 Scoring and ranking...")
                progress_bar.progress(80)
                
                # Step 6: Complete
                status_container.info("✅ Done!")
                progress_bar.progress(100)
                
                # Clear progress indicators
                progress_container.empty()
                status_container.empty()
                
                # Store result in session state
                st.session_state["last_result"] = result
                st.session_state["last_playlist_name"] = result.playlist_name
            
            # Display success message
            st.success(f"✅ Generated {len(result.recommendations)} recommendations for **{result.playlist_name}**!")
            
            # Render recommendations
            render_recommendations(result)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            
            # Show troubleshooting tips
            with st.expander("🔧 Troubleshooting"):
                st.markdown("""
                **Common issues:**
                
                1. **Invalid credentials**: Double-check your Client ID and Secret
                2. **Invalid playlist URL**: Make sure the playlist is public
                3. **API rate limits**: Wait a moment and try again
                4. **Network issues**: Check your internet connection
                
                If the issue persists, try refreshing the page.
                """)
    
    # Show previous results if available
    elif "last_result" in st.session_state:
        st.info(f"Showing previous results for **{st.session_state.get('last_playlist_name', 'Unknown')}**")
        render_recommendations(st.session_state["last_result"])


if __name__ == "__main__":
    main()
