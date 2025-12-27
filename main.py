import streamlit as st
import pickle
from model import Recommender
import pandas as pd
import requests

# ---------------- Load model ----------------
model_path = "model.pkl"

@st.cache_resource
def load_model():
    with open(model_path, "rb") as file:
        return pickle.load(file)

recommender = load_model()
df = recommender.df

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# ---------------- Custom CSS & Styling (Updated) ----------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #0f0c29, #302b63, #24243e);
        color: white;
    }

    /* The Main Card Container */
    .movie-card {
        position: relative;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        overflow: hidden; /* Important to clip the overlay */
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
        height: 450px; /* Fixed height for uniformity */
    }
    
    .movie-card:hover {
        transform: scale(1.03);
        border: 1px solid #00d2ff;
    }

    /* The Poster Image */
    .movie-poster {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    /* The Description Overlay (Hidden by default) */
    .movie-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.85); /* Darker background for text readability */
        color: white;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 20px;
        opacity: 0; /* Hidden */
        transition: opacity 0.4s ease;
        text-align: center;
    }

    /* Reveal overlay on hover */
    .movie-card:hover .movie-overlay {
        opacity: 1;
    }

    .overlay-title {
        color: #00d2ff;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }

    .overlay-desc {
        font-size: 0.9rem;
        line-height: 1.4;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------- Sidebar ----------------
with st.sidebar:
    
    st.markdown("""
        
            <style>
            [data-testid="stSidebar"] {
                background: linear-gradient(to bottom, #0f0c29, #1a1a2e);
                border-right: 1px solid rgba(255, 255, 255, 0.1);
            }
            .sidebar-title {
                color: #00d2ff;
                text-align: center;
                font-family: 'Trebuchet MS', sans-serif;
                letter-spacing: 2px;
                margin-top: 20px;
            }
            .sidebar-subtext {
                color: rgba(255, 255, 255, 0.5);
                text-align: center;
                font-size: 0.8rem;
                margin-bottom: 30px;
            }
        </style>
        <h2 class="sidebar-title">SYSTEM ENGINE</h2>
        <p class="sidebar-subtext">NLP-Analysis</p>
    """, unsafe_allow_html=True)

    
    st.image("https://cdn-icons-png.flaticon.com/512/3658/3658959.png", width=120)

    st.markdown("---")

    
    st.markdown("""
        <div style='padding: 10px; border-radius: 10px; background: rgba(255,255,255,0.03);'>
            <p style='color: #00d2ff; font-size: 14px; margin-bottom: 5px;'>📡 ENGINE STATUS</p>
            <p style='color: #00ff88; font-size: 12px;'>● OPERATIONAL</p>
            <br>
            <p style='color: #00d2ff; font-size: 14px; margin-bottom: 5px;'>🧠 MODEL</p>
            <p style='color: white; font-size: 12px;'>Weighted Cosine Similarity </p>
            <br>
            <p style='color: #00d2ff; font-size: 14px; margin-bottom: 5px;'>📊 DATASET</p>
            <p style='color: white; font-size: 12px;'>TMDB API Scraping</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Bottom accent
    st.caption("AI-Powered Recommendations")


# ---------------- Title ----------------
st.markdown("<h1 style='text-align: center; color: white;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #00d2ff;'>Discover your next favorite film using AI</p>", unsafe_allow_html=True)
st.divider()

# ---------------- Inputs ----------------
col_in1, col_in2, col_in3 = st.columns([4, 1.5, 1.5])

with col_in1:
    movie_title = st.selectbox(
        "Select a movie you like:",
        options=sorted(df["title"].tolist()),
        index=None,
        placeholder="Search for a movie..."
    )

with col_in2:
    num_recommendations = st.number_input(
        "How many suggestions?",
        min_value=1,
        max_value=12,
        value=6,
        step=3
    )

with col_in3:
    st.write(" ")
    st.write(" ")
    run = st.button("🎯 Get Recommendations", use_container_width=True)

# ---------------- Results ----------------
if run:
    if movie_title is None:
        st.warning("Please select a movie first!")
    else:
        with st.spinner("Analyzing our database..."):
            recommendations = recommender.recommend(
                movie_title,
                num_recommendations=num_recommendations
            )

        if isinstance(recommendations, str):
            st.error(recommendations)
        elif recommendations.empty:
            st.warning("No recommendations found.")
        else:
            st.markdown(f"### Suggestions for you:")
            
            for i in range(len(recommendations)):
                if i % 3 == 0:
                    cols = st.columns(3)
                
                with cols[i % 3]:
                    row = recommendations.iloc[i]
                    poster_url = row.get("poster", "https://via.placeholder.com/500x750?text=No+Poster")
                    
                    # HTML Card with Hover Overlay
                    st.markdown(f"""
                        <div class="movie-card">
                            <img src="{poster_url}" class="movie-poster">
                            <div class="movie-overlay">
                                <div class="overlay-title">{row['title']}</div>
                                <div class="movie-meta" style="color: #00d2ff; margin-bottom: 10px;">
                                    ⭐ Score: {row.get('similarity_score', 0):.2f}
                                </div>
                                <div class="overlay-desc">
                                    {row.get('description', 'No description available.')[:200]}...
                                </div>
                                <div style="margin-top: 15px; font-size: 0.8rem; font-style: italic;">
                                    {row.get('genres', '')}
                                </div>
                            </div>
                        </div>

                    """, unsafe_allow_html=True)
