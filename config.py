"""
Configuration for Movie Recommendation System
"""

import os
from pathlib import Path

# --- PATHS ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
USERS_DIR = DATA_DIR / "users"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
USERS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- DATA FILES ---
MOVIES_CSV = DATA_DIR / "movies.csv"
LINKS_CSV = DATA_DIR / "links.csv"  # For TMDB/IMDB integration later
TAGS_CSV = DATA_DIR / "tags.csv"    # Optional enrichment

# --- MODEL FILES ---
COLLABORATIVE_MODEL_PATH = MODELS_DIR / "movie_lstm_v2_trained.pth"
MOVIE_MAPPINGS_PATH = MODELS_DIR / "movie_sequences_v2.pth"

# --- CHROMADB ---
CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")

# --- LLM CONFIGURATION ---
LLM_CONFIG = {
    "model": "llama3.2",
    "temperature": 0,
}

# --- COLLABORATIVE MODEL CONFIG ---
COLLAB_CONFIG = {
    "context_size": 10,  # Number of movies in history for prediction
    "embedding_dim": 256,
    "hidden_dim": 512,
    "num_layers": 2,
    "dropout": 0.4,
}

# --- RL CONFIG ---
RL_CONFIG = {
    "learning_rate": 1e-3,
    "embedding_dim": 64,
    "hidden_dim": 128,
}

# --- RECOMMENDATION SETTINGS ---
RECOMMENDATION_CONFIG = {
    "retrieval_candidates": 100,    # How many to fetch from ChromaDB
    "seed_movies_required": 10,     # Minimum seed movies for new user
    "top_k_display": [5, 10, 20],   # Display tiers
}

# --- GENRE MAPPINGS ---
GENRE_LIST = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

GENRE_TO_ID = {genre: i+1 for i, genre in enumerate(GENRE_LIST)}
GENRE_TO_ID["(no genres listed)"] = 0
ID_TO_GENRE = {v: k for k, v in GENRE_TO_ID.items()}

# --- RATING TO REWARD MAPPING ---
RATING_TO_REWARD = {
    0: -1.0,   # Hated it
    1: -0.5,   # Disliked
    2:  0.0,   # Meh
    3:  0.5,   # Okay
    4:  0.8,   # Liked it
    5:  1.0    # Loved it
}

# --- TMDB CONFIGURATION (for future actor/director support) ---
TMDB_CONFIG = {
    "enabled": False,  # Set to True when you add TMDB API
    "api_key": os.getenv("TMDB_API_KEY", ""),
    "base_url": "https://api.themoviedb.org/3",
}
