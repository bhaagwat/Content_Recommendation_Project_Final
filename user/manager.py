"""
User Profile and History Management
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import USERS_DIR, RECOMMENDATION_CONFIG


class UserManager:
    """Manages user profiles and watch history."""
    
    def __init__(self):
        self.current_user: Optional[str] = None
        self.profile: Optional[Dict] = None
        self.history: Optional[List] = None
    
    def _get_user_dir(self, username: str) -> Path:
        """Get the directory path for a user."""
        return USERS_DIR / username.lower()
    
    def _get_profile_path(self, username: str) -> Path:
        """Get the profile.json path for a user."""
        return self._get_user_dir(username) / "profile.json"
    
    def _get_history_path(self, username: str) -> Path:
        """Get the history.json path for a user."""
        return self._get_user_dir(username) / "history.json"
    
    def _get_rl_weights_path(self, username: str) -> Path:
        """Get the RL weights path for a user."""
        return self._get_user_dir(username) / "rl_weights.pth"
    
    def user_exists(self, username: str) -> bool:
        """Check if a user already exists."""
        return self._get_profile_path(username).exists()
    
    def create_user(self, username: str) -> bool:
        """Create a new user with empty profile."""
        if self.user_exists(username):
            return False
        
        user_dir = self._get_user_dir(username)
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Create empty profile
        profile = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "preferences": {
                "genres": [],
                "actors": [],      # For future TMDB integration
                "directors": [],   # For future TMDB integration
                "keywords": []
            },
            "seed_movies": [],
            "onboarding_complete": False
        }
        
        # Create empty history
        history = {
            "watched": [],
            "total_movies": 0,
            "avg_rating": 0.0
        }
        
        # Save files
        with open(self._get_profile_path(username), 'w') as f:
            json.dump(profile, f, indent=2)
        
        with open(self._get_history_path(username), 'w') as f:
            json.dump(history, f, indent=2)
        
        return True
    
    def login(self, username: str) -> Dict:
        """Login a user and load their data."""
        if not self.user_exists(username):
            raise ValueError(f"User '{username}' does not exist.")
        
        self.current_user = username
        
        # Load profile
        with open(self._get_profile_path(username), 'r') as f:
            self.profile = json.load(f)
        
        # Load history
        with open(self._get_history_path(username), 'r') as f:
            self.history = json.load(f)
        
        return {
            "profile": self.profile,
            "history": self.history
        }
    
    def logout(self):
        """Logout current user."""
        self.current_user = None
        self.profile = None
        self.history = None
    
    def save_profile(self):
        """Save current user's profile."""
        if not self.current_user:
            raise ValueError("No user logged in.")
        
        with open(self._get_profile_path(self.current_user), 'w') as f:
            json.dump(self.profile, f, indent=2)
    
    def save_history(self):
        """Save current user's history."""
        if not self.current_user:
            raise ValueError("No user logged in.")
        
        with open(self._get_history_path(self.current_user), 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def update_preferences(self, preferences: Dict):
        """Update user preferences from onboarding."""
        if not self.profile:
            raise ValueError("No user logged in.")
        
        self.profile["preferences"].update(preferences)
        self.save_profile()
    
    def add_seed_movies(self, movie_ids: List[int]):
        """Add seed movies during onboarding."""
        if not self.profile:
            raise ValueError("No user logged in.")
        
        self.profile["seed_movies"] = movie_ids
        
        # Also add to history as "watched" with neutral rating
        for movie_id in movie_ids:
            self.history["watched"].append({
                "movie_id": movie_id,
                "rating": None,  # No rating for seed movies
                "timestamp": datetime.now().isoformat(),
                "is_seed": True
            })
        
        self.history["total_movies"] = len(self.history["watched"])
        
        # Mark onboarding complete
        if len(movie_ids) >= RECOMMENDATION_CONFIG["seed_movies_required"]:
            self.profile["onboarding_complete"] = True
        
        self.save_profile()
        self.save_history()
    
    def add_watched_movie(self, movie_id: int, movie_title: str, rating: int):
        """Add a movie to watch history with rating."""
        if not self.history:
            raise ValueError("No user logged in.")
        
        # Add to history
        self.history["watched"].append({
            "movie_id": movie_id,
            "title": movie_title,
            "rating": rating,
            "timestamp": datetime.now().isoformat(),
            "is_seed": False
        })
        
        # Update stats
        self.history["total_movies"] = len(self.history["watched"])
        
        # Calculate average rating (excluding seed movies)
        rated_movies = [m for m in self.history["watched"] if m["rating"] is not None]
        if rated_movies:
            self.history["avg_rating"] = sum(m["rating"] for m in rated_movies) / len(rated_movies)
        
        self.save_history()
    
    def get_watch_history_ids(self, limit: int = 10) -> List[int]:
        """Get the last N watched movie IDs for collaborative filtering."""
        if not self.history:
            return []
        
        # Get movie IDs in reverse chronological order
        watched = self.history["watched"]
        movie_ids = [m["movie_id"] for m in reversed(watched)]
        
        return movie_ids[:limit]
    
    def get_all_watched_ids(self) -> set:
        """Get all watched movie IDs (to exclude from recommendations)."""
        if not self.history:
            return set()
        
        return {m["movie_id"] for m in self.history["watched"]}
    
    def is_onboarding_complete(self) -> bool:
        """Check if user has completed onboarding."""
        if not self.profile:
            return False
        return self.profile.get("onboarding_complete", False)
    
    def get_user_stats(self) -> Dict:
        """Get user statistics for display."""
        if not self.profile or not self.history:
            return {}
        
        return {
            "username": self.profile["username"],
            "member_since": self.profile["created_at"][:10],
            "total_movies": self.history["total_movies"],
            "avg_rating": round(self.history["avg_rating"], 1),
            "favorite_genres": self.profile["preferences"].get("genres", [])[:3]
        }
    
    def get_all_users(self) -> List[str]:
        """Get list of all registered users."""
        users = []
        if USERS_DIR.exists():
            for user_dir in USERS_DIR.iterdir():
                if user_dir.is_dir() and (user_dir / "profile.json").exists():
                    users.append(user_dir.name)
        return sorted(users)
    
    def get_rl_weights_path(self) -> Optional[Path]:
        """Get path to current user's RL weights."""
        if not self.current_user:
            return None
        return self._get_rl_weights_path(self.current_user)
