"""
Explainer Agent - Generates explanations for recommendations
"""

from typing import Dict, List, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import LLM_CONFIG, ID_TO_GENRE


class ExplainerAgent:
    """Generates natural language explanations for recommendations."""
    
    def __init__(self):
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the Ollama LLM."""
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.prompts import PromptTemplate
            
            self.llm = ChatOllama(
                model=LLM_CONFIG["model"],
                temperature=0.3  # Low temperature = less hallucination
            )
            
            self.explain_template = PromptTemplate(
                input_variables=["query", "title", "year", "genre", "score", "user_history", "collab_score"],
                template="""You are a movie recommendation explainer. Be concise and factual.

User searched for: "{query}"
User's watched movies: {user_history}

Recommended movie: "{title}" ({year}) - {genre}
Collaborative score: {collab_score}% (based on similar users)
Overall match: {score}%

Rules:
- Write exactly ONE sentence (max 25 words)
- ONLY reference movies from "User's watched movies" list above
- Do NOT invent or assume any movies
- If user history is "None", say "based on your search"
- Focus on: genre match, viewing patterns, or query relevance

Your explanation:"""
            )
            
            self.chain = self.explain_template | self.llm
            print(f"✓ Explainer initialized with {LLM_CONFIG['model']}")
            
        except Exception as e:
            print(f"⚠ Could not initialize Explainer: {e}")
            self.llm = None
    
    def explain(
        self, 
        user_query: str, 
        movie: Dict,
        user_history: List[str] = None
    ) -> str:
        """
        Generate explanation for why a movie was recommended.
        
        Args:
            user_query: What the user searched for
            movie: The recommended movie with metadata and scores
            user_history: List of movie titles user has watched
        
        Returns:
            One sentence explanation
        """
        
        # Extract movie info
        meta = movie.get('metadata', {})
        title = movie.get('title', meta.get('title', 'Unknown'))
        year = meta.get('year', 'N/A')
        genre_id = meta.get('genre_id', 0)
        genre = ID_TO_GENRE.get(genre_id, meta.get('primary_genre', 'General'))
        
        # Extract scores
        collab_score = movie.get('collab_score', 0)
        final_score = movie.get('final_score', collab_score)
        
        # Format user history clearly
        if user_history and len(user_history) > 0:
            # Filter out empty strings and limit to 5
            valid_history = [h for h in user_history if h and h.strip()][:5]
            if valid_history:
                history_str = ", ".join([f'"{h}"' for h in valid_history])
            else:
                history_str = "None"
        else:
            history_str = "None"
        
        # Try LLM explanation
        if self.llm:
            try:
                result = self.chain.invoke({
                    "query": user_query if user_query else "general recommendation",
                    "title": title,
                    "year": year,
                    "genre": genre,
                    "score": int(final_score * 100),
                    "collab_score": f"{collab_score * 100:.2f}",
                    "user_history": history_str
                })
                
                response = result.content.strip()
                
                # Clean up response - remove quotes if wrapped
                if response.startswith('"') and response.endswith('"'):
                    response = response[1:-1]
                
                return response
                
            except Exception as e:
                print(f"    [Explainer] LLM error: {e}")
        
        # Fallback to rule-based explanation
        return self._fallback_explanation(
            title=title,
            genre=genre,
            collab_score=collab_score,
            final_score=final_score,
            user_history=user_history,
            query=user_query
        )
    
    def _fallback_explanation(
        self, 
        title: str, 
        genre: str, 
        collab_score: float,
        final_score: float,
        user_history: List[str] = None,
        query: str = None
    ) -> str:
        """Generate simple factual explanation without LLM."""
        
        match_percent = int(final_score * 100)
        
        # Build explanation based on available info
        if collab_score > 0.02 and user_history and len(user_history) > 0:
            # High collab + has history
            return f"Recommended because fans of \"{user_history[0]}\" often enjoy \"{title}\". ({match_percent}% match)"
        
        elif collab_score > 0.01 and user_history and len(user_history) > 0:
            # Medium collab + has history
            return f"\"{title}\" matches your taste based on your history including \"{user_history[0]}\". ({match_percent}% match)"
        
        elif collab_score > 0.005:
            # Some collab signal
            return f"Users with similar viewing patterns enjoy \"{title}\". ({match_percent}% match)"
        
        elif query:
            # Use query as reason
            return f"\"{title}\" is a great {genre} pick for \"{query}\". ({match_percent}% match)"
        
        else:
            # Generic fallback
            return f"\"{title}\" is a popular {genre} film you might enjoy. ({match_percent}% match)"
    
    def explain_scores(self, movie: Dict) -> str:
        """
        Generate explanation of the score breakdown.
        
        Args:
            movie: Movie with collab_score, rl_score, final_score
        
        Returns:
            Score breakdown explanation
        """
        
        title = movie.get('title', 'This movie')
        collab = movie.get('collab_score', 0)
        rl = movie.get('rl_score', 0.5)
        final = movie.get('final_score', 0)
        
        parts = []
        
        # Collaborative score explanation
        if collab > 0.03:
            parts.append(f"Strong pattern match ({collab*100:.1f}%)")
        elif collab > 0.01:
            parts.append(f"Good pattern match ({collab*100:.1f}%)")
        elif collab > 0.001:
            parts.append(f"Weak pattern match ({collab*100:.2f}%)")
        else:
            parts.append(f"Minimal pattern ({collab*100:.3f}%)")
        
        # RL score explanation
        if rl > 0.6:
            parts.append(f"High personal fit ({rl*100:.0f}%)")
        elif rl > 0.4:
            parts.append(f"Moderate personal fit ({rl*100:.0f}%)")
        else:
            parts.append(f"Low personal fit ({rl*100:.0f}%)")
        
        return f"{title}: {' • '.join(parts)} → {final*100:.0f}% overall"
    
    def get_recommendation_reasons(
        self, 
        movie: Dict, 
        user_history: List[str] = None
    ) -> List[str]:
        """
        Get list of reasons why movie was recommended.
        
        Returns:
            List of reason strings
        """
        
        reasons = []
        
        meta = movie.get('metadata', {})
        genre = meta.get('primary_genre', 'Unknown')
        collab = movie.get('collab_score', 0)
        rl = movie.get('rl_score', 0.5)
        
        # Collaborative reasoning
        if collab > 0.02:
            reasons.append("Users with similar taste frequently watch this")
        elif collab > 0.005:
            reasons.append("Matches viewing patterns of similar users")
        
        # RL reasoning
        if rl > 0.6:
            reasons.append("Highly aligned with your rating preferences")
        elif rl > 0.5:
            reasons.append("Fits your preference patterns")
        
        # History reasoning
        if user_history and len(user_history) > 0:
            reasons.append(f"Related to movies you've watched like \"{user_history[0]}\"")
        
        # Genre
        reasons.append(f"Popular choice in {genre} genre")
        
        return reasons[:3]  # Return top 3 reasons
