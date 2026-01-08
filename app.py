"""
Movie Recommendation System - Clean UI
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import RECOMMENDATION_CONFIG, USERS_DIR
from user.manager import UserManager
from agents.parser import ParserAgent
from agents.retriever import RetrieverEngine
from agents.explainer import ExplainerAgent
from models.collaborative import CollaborativeFilter
from models.rl_optimizer import RLOptimizer


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #1a1a2e;
        color: #ffffff;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        color: #ffffff;
        margin-bottom: 0;
        letter-spacing: -1px;
    }
    
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #888888;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .card {
        background-color: #252545;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .user-card {
        background-color: #2d2d4a;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .user-stat {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ffffff;
    }
    
    .user-label {
        font-size: 0.8rem;
        color: #888888;
        text-transform: uppercase;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background-color: #3d3d5c;
        border: none;
        border-radius: 8px;
        color: #ffffff;
        padding: 1rem;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #3d3d5c;
        border: none;
        border-radius: 8px;
        color: #ffffff;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #4a4a6a;
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #5a5a7a;
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #6c63ff;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #5a52d9;
    }
    
    /* Movie card */
    .movie-card {
        background-color: #2d2d4a;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .movie-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
    }
    
    .movie-info {
        font-size: 0.85rem;
        color: #888888;
    }
    
    .match-score {
        font-size: 0.9rem;
        color: #6c63ff;
        font-weight: 600;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #888888;
        border: none;
        padding: 0.5rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #ffffff;
        border-bottom: 2px solid #6c63ff;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #6c63ff;
    }
    
    /* Divider */
    hr {
        border-color: #3d3d5c;
    }
    
    /* Select box */
    .stSelectbox > div > div {
        background-color: #3d3d5c;
        border: none;
        border-radius: 8px;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
    }
    
    /* Info text */
    .info-text {
        color: #888888;
        font-size: 0.9rem;
    }
    
    /* Bottom stats */
    .bottom-stats {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #1a1a2e;
        border-top: 1px solid #3d3d5c;
        padding: 0.5rem 2rem;
        font-size: 0.75rem;
        color: #666666;
    }
</style>
""", unsafe_allow_html=True)


# --- INITIALIZE SESSION STATE ---
def init_session_state():
    if 'user_manager' not in st.session_state:
        st.session_state.user_manager = UserManager()
    
    if 'parser' not in st.session_state:
        st.session_state.parser = ParserAgent()
    
    if 'retriever' not in st.session_state:
        st.session_state.retriever = RetrieverEngine()
        st.session_state.retriever.seed_database()
    
    if 'collab_filter' not in st.session_state:
        st.session_state.collab_filter = CollaborativeFilter()
    
    if 'explainer' not in st.session_state:
        st.session_state.explainer = ExplainerAgent()
    
    if 'rl_optimizer' not in st.session_state:
        st.session_state.rl_optimizer = None
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'current_recommendations' not in st.session_state:
        st.session_state.current_recommendations = []
    
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    
    if 'selected_seed_movies' not in st.session_state:
        st.session_state.selected_seed_movies = []
    
    if 'onboarding_candidates' not in st.session_state:
        st.session_state.onboarding_candidates = []


init_session_state()


# --- HELPER FUNCTIONS ---
def login_user(username: str):
    um = st.session_state.user_manager
    try:
        um.login(username)
        st.session_state.logged_in = True
        rl_path = um.get_rl_weights_path()
        st.session_state.rl_optimizer = RLOptimizer(rl_path)
        return True
    except ValueError:
        return False


def logout_user():
    st.session_state.user_manager.logout()
    st.session_state.logged_in = False
    st.session_state.rl_optimizer = None
    st.session_state.current_recommendations = []
    st.session_state.current_query = ""
    st.session_state.selected_seed_movies = []


def create_new_user(username: str):
    um = st.session_state.user_manager
    if um.user_exists(username):
        return False, "Username already exists"
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if not username.isalnum():
        return False, "Username must be alphanumeric"
    um.create_user(username)
    return True, "Account created!"


# --- LOGIN PAGE ---
def render_login_page():
    st.markdown('<h1 class="main-title">MOVIE RECOMMENDER</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">USING LSTM, RL AND SEMANTIC SEARCH</p>', unsafe_allow_html=True)
    
    # Centered login box
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["LOGIN", "REGISTER"])
        
        with tab1:
            um = st.session_state.user_manager
            existing_users = um.get_all_users()
            
            if existing_users:
                selected_user = st.selectbox(
                    "Select user",
                    options=[""] + existing_users,
                    key="login_select",
                    label_visibility="collapsed"
                )
                
                st.write("")  # Spacer
                
                if st.button("LOGIN", key="login_btn", type="primary", use_container_width=True):
                    if selected_user:
                        if login_user(selected_user):
                            st.rerun()
                        else:
                            st.error("Login failed")
                    else:
                        st.warning("Please select a user")
            else:
                st.info("No users yet. Please register!")
        
        with tab2:
            new_username = st.text_input(
                "Username",
                key="register_input",
                max_chars=20,
                placeholder="Enter username"
            )
            
            st.write("")  # Spacer
            
            if st.button("CREATE ACCOUNT", key="register_btn", type="primary", use_container_width=True):
                if new_username:
                    success, msg = create_new_user(new_username.lower())
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
        
        st.markdown('</div>', unsafe_allow_html=True)


# --- USER INFO CARD ---
def render_user_card():
    um = st.session_state.user_manager
    stats = um.get_user_stats()
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'<p class="user-label">USERNAME</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="user-stat">{stats["username"]}</p>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<p class="user-label">MOVIES WATCHED</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="user-stat">{stats["total_movies"]}</p>', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'<p class="user-label">AVG RATING</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="user-stat">{stats["avg_rating"]} ⭐</p>', unsafe_allow_html=True)
        
        with col4:
            if st.button("LOGOUT", key="logout_btn"):
                logout_user()
                st.rerun()


# --- ONBOARDING PAGE ---
def render_onboarding():
    um = st.session_state.user_manager
    
    # Header with user card
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="section-header">PERSONALISE YOUR EXPERIENCE</h1>', unsafe_allow_html=True)
        st.markdown('<p class="info-text">Tell us about yourself</p>', unsafe_allow_html=True)
    with col2:
        render_user_card_small()
    
    st.markdown("---")
    
    # Step 1: Describe preferences
    if not st.session_state.onboarding_candidates:
        st.markdown('<h2 class="section-header">DESCRIBE YOUR TASTE</h2>', unsafe_allow_html=True)
        
        preference_input = st.text_area(
            "What movies do you enjoy?",
            placeholder="Example: I love The Dark Knight, Fight Club, Pulp Fiction. I enjoy thrillers and drama.",
            height=100,
            key="preference_input",
            label_visibility="collapsed"
        )
        
        st.write("")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("FIND MOVIES FOR ME!", key="find_movies_btn", type="primary", use_container_width=True):
                if preference_input:
                    parser = st.session_state.parser
                    preferences = parser.parse_preferences(preference_input)
                    um.update_preferences(preferences)
                    
                    retriever = st.session_state.retriever
                    collab = st.session_state.collab_filter
                    
                    candidates = retriever.search_by_preferences(preferences, k=100, popular_only=True)
                    
                    if collab.is_loaded and collab.movie_to_idx:
                        candidates = [c for c in candidates if c['id'] in collab.movie_to_idx]
                    
                    candidates = candidates[:30]
                    st.session_state.onboarding_candidates = candidates
                    st.rerun()
                else:
                    st.warning("Please describe your taste first")
    
    # Step 2: Select seed movies
    else:
        st.markdown('<h2 class="section-header">SELECT 10 MOVIES YOU WATCHED AND LIKED</h2>', unsafe_allow_html=True)
        
        candidates = st.session_state.onboarding_candidates
        selected = st.session_state.selected_seed_movies.copy()
        
        # Movie grid
        cols = st.columns(5)
        
        for i, movie in enumerate(candidates):
            col = cols[i % 5]
            
            with col:
                is_selected = movie['id'] in [m['id'] for m in selected]
                
                with st.container():
                    st.markdown(f"**{movie['title'][:25]}...**" if len(movie['title']) > 25 else f"**{movie['title']}**")
                    st.caption(f"{movie['metadata'].get('primary_genre', 'N/A')} • {movie['metadata'].get('year', 'N/A')}")
                    
                    if is_selected:
                        if st.button("✓ SELECTED", key=f"sel_{movie['id']}", type="primary"):
                            selected = [m for m in selected if m['id'] != movie['id']]
                            st.session_state.selected_seed_movies = selected
                            st.rerun()
                    else:
                        if st.button("SELECT", key=f"sel_{movie['id']}"):
                            if len(selected) < 15:
                                selected.append(movie)
                                st.session_state.selected_seed_movies = selected
                                st.rerun()
                
                st.write("")
        
        # Selection count and confirm
        st.markdown("---")
        
        num_selected = len(st.session_state.selected_seed_movies)
        required = RECOMMENDATION_CONFIG['seed_movies_required']
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"**Selected: {num_selected} / {required} movies**")
            
            if num_selected >= required:
                if st.button("✓ COMPLETE SETUP", key="complete_onboarding", type="primary", use_container_width=True):
                    seed_ids = [m['id'] for m in st.session_state.selected_seed_movies]
                    um.add_seed_movies(seed_ids)
                    st.session_state.selected_seed_movies = []
                    st.session_state.onboarding_candidates = []
                    st.success("🎉 Setup complete!")
                    st.rerun()
            else:
                st.info(f"Please select at least {required} movies")


def render_user_card_small():
    um = st.session_state.user_manager
    stats = um.get_user_stats()
    
    st.markdown(f"""
    <div class="user-card">
        <p class="user-stat">{stats["username"]}</p>
        <p class="user-label">AVG RATING: {stats["avg_rating"]} ⭐</p>
        <p class="user-label">MOVIES: {stats["total_movies"]}</p>
    </div>
    """, unsafe_allow_html=True)


# --- MAIN RECOMMENDATION PAGE ---
def render_recommendations():
    um = st.session_state.user_manager
    collab = st.session_state.collab_filter
    
    st.markdown('<h1 class="section-header">WHAT DO YOU WANNA WATCH?</h1>', unsafe_allow_html=True)
    
    # Search input
    query = st.text_input(
        "Search",
        placeholder="Example: thriller movie, sci-fi action, comedy from 90s",
        key="search_query",
        label_visibility="collapsed"
    )
    
    st.write("")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_clicked = st.button("FIND MOVIES FOR ME!", key="search_btn", type="primary", use_container_width=True)
    
    if search_clicked and query:
        with st.spinner("Finding movies for you..."):
            parser = st.session_state.parser
            filters = parser.parse_search_query(query)
            
            retriever = st.session_state.retriever
            watched_ids = um.get_all_watched_ids()
            candidates = retriever.search(filters, k=500, exclude_ids=watched_ids)
            
            if not candidates:
                st.warning("No movies found. Try different keywords.")
                return
            
            if collab.is_loaded and collab.movie_to_idx:
                candidates = [c for c in candidates if c['id'] in collab.movie_to_idx]
                
                if len(candidates) == 0:
                    st.warning("No movies found. Try different keywords.")
                    return
            
            candidates = candidates[:100]
            
            watch_history = um.get_watch_history_ids(limit=10)
            
            if watch_history and collab.is_loaded:
                candidates = collab.score_candidates(watch_history, candidates)
            else:
                for c in candidates:
                    c['collab_score'] = 0.5
            
            rl = st.session_state.rl_optimizer
            if rl:
                is_new_user = len(watch_history) < 20
                candidates = rl.rank_candidates(candidates, is_new_user=is_new_user)
            else:
                for c in candidates:
                    c['rl_score'] = 0.5
                    c['final_score'] = c.get('collab_score', 0.5)
            
            st.session_state.current_recommendations = candidates
            st.session_state.current_query = query
    
    # Display recommendations
    if st.session_state.current_recommendations:
        render_recommendation_results()


def render_recommendation_results():
    candidates = st.session_state.current_recommendations
    
    st.markdown("---")
    
    # Tabs for Top 5, Top 10, Top 20
    tab1, tab2, tab3 = st.tabs(["TOP 5", "TOP 10", "TOP 20"])
    
    with tab1:
        render_movie_list(candidates[:5], "top5")
    
    with tab2:
        render_movie_list(candidates[:10], "top10")
    
    with tab3:
        render_movie_list(candidates[:20], "top20")


def render_movie_list(movies: list, tier: str):
    um = st.session_state.user_manager
    
    if not movies:
        st.info("No movies to display.")
        return
    
    # Get user history for explanations
    history = um.history.get('watched', [])
    history_titles = [h.get('title', '') for h in history if h.get('title')]
    
    for i, movie in enumerate(movies):
        col1, col2, col3 = st.columns([4, 1, 1])
        
        with col1:
            title = movie.get('title', movie.get('metadata', {}).get('title', 'Unknown'))
            year = movie.get('metadata', {}).get('year', 'N/A')
            genre = movie.get('metadata', {}).get('primary_genre', 'N/A')
            
            st.markdown(f"**{i+1}. {title}**")
            st.caption(f"{genre} • {year}")
            
            # Score bar
            final_score = movie.get('final_score', 0)
            st.progress(min(final_score * 2, 1.0))
            
            # Display all three scores
            collab_score = movie.get('collab_score', 0)
            rl_score = movie.get('rl_score', 0.5)
            
            st.caption(
                f"Match: **{int(final_score * 100)}%** · "
                f"Collab: {collab_score:.3f} · "
                f"RL: {rl_score:.2f}"
            )
        
        with col2:
            if st.button("🎬 WATCH", key=f"watch_{tier}_{movie['id']}"):
                st.session_state.watching_movie = movie
                st.session_state.show_rating_dialog = True
                st.rerun()
        
        with col3:
            if st.button("WHY?", key=f"why_{tier}_{movie['id']}"):
                explainer = st.session_state.explainer
                query = st.session_state.get('current_query', '')
                
                explanation = explainer.explain(
                    user_query=query,
                    movie=movie,
                    user_history=history_titles
                )
                st.info(explanation)
        
        st.markdown("---")

# --- RATING DIALOG ---
def render_rating_dialog():
    if not st.session_state.get('show_rating_dialog'):
        return
    
    movie = st.session_state.get('watching_movie')
    if not movie:
        return
    
    um = st.session_state.user_manager
    rl = st.session_state.rl_optimizer
    
    st.markdown("---")
    st.markdown('<h2 class="section-header">⭐ RATE THIS MOVIE</h2>', unsafe_allow_html=True)
    
    title = movie.get('title', movie.get('metadata', {}).get('title', 'Unknown'))
    st.write(f"How did you like **{title}**?")
    
    rating = st.slider("Your rating", 0, 5, 3, key="rating_slider", label_visibility="collapsed")
    st.write("⭐" * rating + "☆" * (5 - rating))
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("SUBMIT RATING", key="submit_rating", type="primary", use_container_width=True):
            um.add_watched_movie(movie['id'], title, rating)
            if rl:
                rl.update(movie, rating)
            st.session_state.show_rating_dialog = False
            st.session_state.watching_movie = None
            st.session_state.current_recommendations = []
            st.success("Thanks for rating!")
            st.rerun()
    
    with col2:
        if st.button("SKIP", key="skip_rating", use_container_width=True):
            st.session_state.show_rating_dialog = False
            st.session_state.watching_movie = None
            st.rerun()


# --- BOTTOM STATS ---
def render_bottom_stats():
    collab = st.session_state.collab_filter
    retriever = st.session_state.retriever
    
    db_count = retriever.collection.count() if retriever.collection else 0
    vocab_size = len(collab.movie_to_idx) if collab.is_loaded else 0
    
    st.markdown(f"""
    <div class="bottom-stats">
        Database: {db_count:,} movies | LSTM Vocabulary: {vocab_size:,} | Model: {'✓ Loaded' if collab.is_loaded else '✗ Not loaded'}
    </div>
    """, unsafe_allow_html=True)


# --- MAIN ---
def main():
    if not st.session_state.logged_in:
        render_login_page()
    else:
        um = st.session_state.user_manager
        
        if not um.is_onboarding_complete():
            render_onboarding()
        else:
            # User card at top right
            col1, col2 = st.columns([3, 1])
            with col2:
                render_user_card_small()
            
            render_recommendations()
            render_rating_dialog()
    
    # Bottom stats
    render_bottom_stats()


if __name__ == "__main__":
    main()
