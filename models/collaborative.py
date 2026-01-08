"""
Collaborative Filtering Model - LSTM-based sequence prediction

This wraps the trained LSTM model for use in the recommendation pipeline.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import COLLABORATIVE_MODEL_PATH, MOVIE_MAPPINGS_PATH, COLLAB_CONFIG


class Attention(nn.Module):
    """Attention mechanism for LSTM output."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(lstm_output * attention_weights, dim=1)
        return context, attention_weights


class MovieLSTMv2(nn.Module):
    """
    Bidirectional LSTM with Attention for movie sequence prediction.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int = 256, 
                 hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.4):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.attention = Attention(hidden_dim * 2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.layer_norm(lstm_out)
        context, _ = self.attention(lstm_out)
        logits = self.fc(context)
        return logits


class CollaborativeFilter:
    """
    Wrapper for the trained LSTM model to score movie candidates
    based on user's watch history.
    """
    
    def __init__(self):
        self.model = None
        self.movie_to_idx = {}
        self.idx_to_movie = {}
        self.movie_info = {}
        self.context_size = COLLAB_CONFIG["context_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained LSTM model and mappings."""
        
        # Check if model file exists
        if not COLLABORATIVE_MODEL_PATH.exists():
            print(f"⚠ Collaborative model not found: {COLLABORATIVE_MODEL_PATH}")
            return
        
        try:
            # Load model checkpoint
            checkpoint = torch.load(COLLABORATIVE_MODEL_PATH, map_location=self.device)
            
            # Load mappings from SEPARATE file
            if MOVIE_MAPPINGS_PATH.exists():
                mappings = torch.load(MOVIE_MAPPINGS_PATH, map_location=self.device)
                self.movie_to_idx = mappings.get('movie_to_idx', {})
                self.idx_to_movie = mappings.get('idx_to_movie', {})
                self.movie_info = mappings.get('movie_info', {})
                self.context_size = mappings.get('context_size', 10)
                print(f"✓ Loaded mappings: {len(self.movie_to_idx)} movies")
            else:
                print(f"⚠ Mappings file not found: {MOVIE_MAPPINGS_PATH}")
                return
            
            vocab_size = checkpoint.get('vocab_size', len(self.movie_to_idx) + 1)
            
            # Get config from checkpoint or mappings
            config = checkpoint.get('config', mappings.get('config', {}))
            
            # Initialize model
            self.model = MovieLSTMv2(
                vocab_size=vocab_size,
                embedding_dim=config.get('embedding_dim', COLLAB_CONFIG['embedding_dim']),
                hidden_dim=config.get('hidden_dim', COLLAB_CONFIG['hidden_dim']),
                num_layers=config.get('num_layers', COLLAB_CONFIG['num_layers']),
                dropout=config.get('dropout', COLLAB_CONFIG['dropout'])
            ).to(self.device)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.is_loaded = True
            print(f"✓ Collaborative model loaded. Vocab: {vocab_size}, Context: {self.context_size}")
            
        except Exception as e:
            print(f"⚠ Failed to load collaborative model: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
    
    def score_candidates(
        self, 
        watch_history: List[int], 
        candidates: List[Dict]
    ) -> List[Dict]:
        """
        Score candidate movies based on user's watch history.
        """
        # If model not loaded, return neutral scores
        if not self.is_loaded or not self.model:
            for c in candidates:
                c['collab_score'] = 0.5
            return candidates
        
        # Check how many history movies are in vocabulary
        valid_history = [mid for mid in watch_history if mid in self.movie_to_idx]
        
        # If less than 3 valid movies, LSTM can't make good predictions
        if len(valid_history) < 3:
            print(f"    [Collab] Only {len(valid_history)} movies in vocab. Using fallback.")
            for c in candidates:
                c['collab_score'] = 0.001  # Low score, let other signals dominate
            return candidates
        
        # Pad or trim to context size
        if len(valid_history) < self.context_size:
            padding_needed = self.context_size - len(valid_history)
            valid_history = [0] * padding_needed + valid_history
        else:
            valid_history = valid_history[-self.context_size:]
        
        # Convert to model indices
        history_indices = [self.movie_to_idx.get(mid, 0) for mid in valid_history]
        
        # Create tensor
        context_tensor = torch.tensor([history_indices], dtype=torch.long).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(context_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()
        
        # Score each candidate
        for candidate in candidates:
            movie_id = candidate['id']
            idx = self.movie_to_idx.get(movie_id, 0)
            
            if idx > 0 and idx < len(probs):
                candidate['collab_score'] = probs[idx].item()
            else:
                # Movie not in vocabulary
                candidate['collab_score'] = 0.001
        
        # Sort by collaborative score
        candidates.sort(key=lambda x: x['collab_score'], reverse=True)
        
        return candidates
    def check_movies_in_vocab(self, movie_ids: List[int]) -> dict:
        """Check which movies are in LSTM vocabulary."""
        known = []
        unknown = []
        
        for mid in movie_ids:
            if mid in self.movie_to_idx:
                known.append(mid)
            else:
                unknown.append(mid)
        
        return {
            "known": known,
            "unknown": unknown,
            "known_count": len(known),
            "total": len(movie_ids),
            "coverage": len(known) / len(movie_ids) if movie_ids else 0
        }
    
    def predict_next_movies(
        self, 
        watch_history: List[int], 
        top_k: int = 20
    ) -> List[Dict]:
        """
        Predict the most likely next movies given watch history.
        
        Returns top-k predictions with scores.
        """
        if not self.is_loaded or not self.model:
            return []
        
        if len(watch_history) < self.context_size:
            padding_needed = self.context_size - len(watch_history)
            watch_history = [0] * padding_needed + watch_history
        else:
            watch_history = watch_history[-self.context_size:]
        
        # Convert to indices
        history_indices = [self.movie_to_idx.get(m, 0) for m in watch_history]
        context_tensor = torch.tensor([history_indices], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            logits = self.model(context_tensor)
            probs = torch.softmax(logits, dim=1)
            top_probs, top_indices = probs.topk(top_k, dim=1)
        
        predictions = []
        for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
            movie_id = self.idx_to_movie.get(int(idx))
            if movie_id and movie_id in self.movie_info:
                predictions.append({
                    'id': movie_id,
                    'title': self.movie_info[movie_id]['title'],
                    'genres': self.movie_info[movie_id]['genres'],
                    'collab_score': float(prob)
                })
        
        return predictions
    
    def get_movie_info(self, movie_id: int) -> Optional[Dict]:
        """Get movie info from the model's vocabulary."""
        if movie_id in self.movie_info:
            return self.movie_info[movie_id]
        return None
    
    def is_movie_known(self, movie_id: int) -> bool:
        """Check if a movie is in the collaborative model's vocabulary."""
        return movie_id in self.movie_to_idx
