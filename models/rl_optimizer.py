"""
RL Optimizer - Per-user personalization model

This model learns user-specific preferences on top of the collaborative filter.
Each user has their own weights that adjust recommendation scores.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import RL_CONFIG, RATING_TO_REWARD, GENRE_TO_ID


class UserPreferenceNet(nn.Module):
    """
    Small neural network that learns user-specific adjustments.
    
    Takes movie features and outputs a preference score adjustment.
    """
    
    def __init__(
        self, 
        num_genres: int = 20,
        embedding_dim: int = 64,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Genre embedding
        self.genre_embed = nn.Embedding(num_genres, embedding_dim)
        
        # Content vector projection (from 384-dim sentence transformer)
        self.content_projection = nn.Linear(384, embedding_dim)
        
        # MLP for preference scoring
        # Input: genre_embed(64) + content(64) + stats(2) = 130
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, genre_id, content_vec, stats):
        """
        Args:
            genre_id: [batch] genre indices
            content_vec: [batch, 384] content embeddings
            stats: [batch, 2] popularity and rating stats
        
        Returns:
            [batch, 1] preference scores
        """
        g_embed = self.genre_embed(genre_id)  # [batch, 64]
        c_proj = torch.relu(self.content_projection(content_vec))  # [batch, 64]
        
        x = torch.cat([g_embed, c_proj, stats], dim=-1)  # [batch, 130]
        return self.mlp(x)


class RLOptimizer:
    """
    Manages per-user RL models for personalization.
    """
    
    def __init__(self, user_weights_path: Optional[Path] = None):
        """
        Args:
            user_weights_path: Path to user's personal RL weights
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = user_weights_path
        
        # Initialize model
        self.model = UserPreferenceNet(
            num_genres=len(GENRE_TO_ID) + 1,
            embedding_dim=RL_CONFIG['embedding_dim'],
            hidden_dim=RL_CONFIG['hidden_dim']
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=RL_CONFIG['learning_rate']
        )
        self.criterion = nn.MSELoss()
        
        # Load existing weights if available
        self._load_weights()
        
        # Training history
        self.training_history = []
    
    def _load_weights(self):
        """Load user's saved weights if they exist."""
        if self.weights_path and self.weights_path.exists():
            try:
                state = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state['model_state_dict'])
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
                self.training_history = state.get('training_history', [])
                print(f"✓ Loaded personal RL weights ({len(self.training_history)} updates)")
            except Exception as e:
                print(f"⚠ Could not load RL weights: {e}")
    
    def _save_weights(self):
        """Save user's weights."""
        if self.weights_path:
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_history': self.training_history
            }, self.weights_path)
    
    def set_user_weights_path(self, path: Path):
        """Set the weights path for current user."""
        self.weights_path = path
        self._load_weights()
    
    def score_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Add RL preference scores to candidates.
        
        Args:
            candidates: List of movies with metadata and embeddings
        
        Returns:
            Candidates with 'rl_score' added
        """
        self.model.eval()
        
        for movie in candidates:
            meta = movie.get('metadata', {})
            
            # Prepare inputs
            genre_id = torch.tensor([meta.get('genre_id', 0)]).to(self.device)
            
            # Content vector (from retriever embedding)
            embedding = movie.get('embedding')
            if embedding is not None:
                if len(embedding) != 384:
                    # Pad or truncate to 384
                    embedding = embedding[:384] if len(embedding) > 384 else embedding + [0] * (384 - len(embedding))
                content_vec = torch.tensor([embedding], dtype=torch.float32).to(self.device)
            else:
                content_vec = torch.zeros(1, 384).to(self.device)
            
            # Stats (normalized)
            popularity = meta.get('popularity', 0)
            vote_avg = meta.get('vote_avg', 0)
            stats = torch.tensor([[popularity, vote_avg]], dtype=torch.float32).to(self.device)
            
            # Get score
            with torch.no_grad():
                score = self.model(genre_id, content_vec, stats).item()
            
            movie['rl_score'] = score
        
        return candidates
    
    def rank_candidates(
        self, 
        candidates: List[Dict],
        collab_weight: float = 0.5,
        rl_weight: float = 0.3,
        content_weight: float = 0.2,
        is_new_user: bool = False
    ) -> List[Dict]:
        
        # For new users, trust semantic search more than LSTM
        if is_new_user:
            collab_weight = 0.2
            rl_weight = 0.1
            content_weight = 0.7
        """
        Combine collaborative and RL scores to rank candidates.
        
        Final score = collab_weight * collab_score + rl_weight * rl_score + content_weight * content_score
        """
        
        # Add RL scores
        candidates = self.score_candidates(candidates)
        
        for movie in candidates:
            collab_score = movie.get('collab_score', 0.5)
            rl_score = movie.get('rl_score', 0.5)
            
            # Content score - use retriever similarity if available
            meta = movie.get('metadata', {})
            # Retriever already ranked by semantic similarity
            # Use position-based score (first = best match)
            position = candidates.index(movie) if movie in candidates else 50
            retriever_score = max(0, 1 - (position / 100))  # 1st place = 1.0, 100th = 0.0
            content_score = retriever_score
           
            # Combined score
            movie['final_score'] = (
                collab_weight * collab_score +
                rl_weight * rl_score +
                content_weight * content_score
            )
        
        # Sort by final score
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        return candidates
    
    def update(self, movie: Dict, rating: int):
        """
        Update model based on user's rating.
        
        Args:
            movie: The movie that was rated
            rating: User's rating (0-5)
        """
        self.model.train()
        
        # Convert rating to reward
        reward = RATING_TO_REWARD.get(rating, 0.0)
        # Normalize to 0-1 range for sigmoid output
        target = (reward + 1) / 2  # Maps -1..1 to 0..1
        
        meta = movie.get('metadata', {})
        
        # Prepare inputs
        genre_id = torch.tensor([meta.get('genre_id', 0)]).to(self.device)
        
        embedding = movie.get('embedding')
        if embedding is not None:
            if len(embedding) != 384:
                embedding = embedding[:384] if len(embedding) > 384 else embedding + [0] * (384 - len(embedding))
            content_vec = torch.tensor([embedding], dtype=torch.float32).to(self.device)
        else:
            content_vec = torch.zeros(1, 384).to(self.device)
        
        stats = torch.tensor([
            [meta.get('popularity', 0), meta.get('vote_avg', 0)]
        ], dtype=torch.float32).to(self.device)
        
        target_tensor = torch.tensor([[target]], dtype=torch.float32).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(genre_id, content_vec, stats)
        loss = self.criterion(output, target_tensor)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Record history
        self.training_history.append({
            'movie_id': movie.get('id'),
            'rating': rating,
            'reward': reward,
            'loss': loss.item()
        })
        
        # Save weights
        self._save_weights()
        
        print(f"--- [RL] Updated model | Rating: {rating} | Loss: {loss.item():.4f} ---")
        
        return loss.item()
    
    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        if not self.training_history:
            return {
                'total_updates': 0,
                'avg_loss': 0,
                'avg_rating': 0
            }
        
        recent = self.training_history[-50:]  # Last 50 updates
        
        return {
            'total_updates': len(self.training_history),
            'avg_loss': np.mean([h['loss'] for h in recent]),
            'avg_rating': np.mean([h['rating'] for h in recent]),
            'recent_updates': len(recent)
        }
