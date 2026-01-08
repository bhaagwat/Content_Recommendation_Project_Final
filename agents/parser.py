"""
Parser Agent - Extracts structured filters from natural language using Ollama
"""

import re
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import LLM_CONFIG, GENRE_LIST

# Define allowed genres
ALLOWED_GENRES = Literal[
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "Any"
]


class MoviePreferences(BaseModel):
    """Structured output for user preference parsing."""
    genres: List[str] = Field(
        description="List of preferred genres from: Action, Adventure, Animation, Children, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western",
        default=[]
    )
    actors: List[str] = Field(
        description="List of actor names mentioned",
        default=[]
    )
    directors: List[str] = Field(
        description="List of director names mentioned",
        default=[]
    )
    keywords: List[str] = Field(
        description="Key themes or moods (e.g., 'suspenseful', 'romantic', 'mind-bending')",
        default=[]
    )


class MovieFilter(BaseModel):
    """Structured output for search query parsing."""
    genre: str = Field(
        description="The most relevant primary genre",
        default="Any"
    )
    year_start: int = Field(
        description="Start year filter",
        default=1900
    )
    year_end: int = Field(
        description="End year filter", 
        default=2025
    )
    keyword: str = Field(
        description="Central theme or keyword for semantic search"
    )
    mood: Optional[str] = Field(
        description="Mood/tone (e.g., 'dark', 'uplifting', 'intense')",
        default=None
    )


class ParserAgent:
    """Parses natural language into structured movie filters using Ollama."""
    
    def __init__(self):
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the Ollama LLM."""
        try:
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                model=LLM_CONFIG["model"],
                temperature=LLM_CONFIG["temperature"]
            )
            print(f"✓ Parser initialized with {LLM_CONFIG['model']}")
        except Exception as e:
            print(f"⚠ Could not initialize Ollama: {e}")
            print("  Parser will use fallback regex-based extraction")
            self.llm = None
    
    def parse_preferences(self, user_input: str) -> Dict:
        """
        Parse user's preference description during onboarding.
        
        Example: "I like Christopher Nolan movies, sci-fi, and Leonardo DiCaprio"
        Returns: {genres: ["Sci-Fi"], actors: ["Leonardo DiCaprio"], directors: ["Christopher Nolan"]}
        """
        print(f"--- [Parser] Analyzing preferences: '{user_input}' ---")
        
        if self.llm:
            try:
                structured_llm = self.llm.with_structured_output(MoviePreferences)
                result = structured_llm.invoke(
                    f"Extract movie preferences from this text. "
                    f"Identify genres, actor names, director names, and keywords/themes: {user_input}"
                )
                preferences = result.dict()
                
                # Validate genres
                preferences["genres"] = [
                    g for g in preferences["genres"] 
                    if g in GENRE_LIST
                ]
                
                print(f"    Extracted: {preferences}")
                return preferences
                
            except Exception as e:
                print(f"    LLM Error: {e}. Using fallback.")
        
        # Fallback: regex-based extraction
        return self._fallback_parse_preferences(user_input)
    
    def _fallback_parse_preferences(self, user_input: str) -> Dict:
        """Regex-based fallback for preference parsing."""
        preferences = {
            "genres": [],
            "actors": [],
            "directors": [],
            "keywords": []
        }
        
        input_lower = user_input.lower()
        
        # Extract genres
        for genre in GENRE_LIST:
            if genre.lower() in input_lower:
                preferences["genres"].append(genre)
        
        # Extract common keywords
        mood_keywords = ["dark", "funny", "scary", "romantic", "thrilling", 
                        "intense", "uplifting", "mind-bending", "emotional"]
        for keyword in mood_keywords:
            if keyword in input_lower:
                preferences["keywords"].append(keyword)
        
        return preferences
    
    def parse_search_query(self, user_input: str) -> Dict:
        """
        Parse a search query for movie recommendations.
        
        Example: "I want a scary movie from the 90s"
        Returns: {genre: "Horror", year_start: 1990, year_end: 1999, keyword: "scary"}
        """
        print(f"--- [Parser] Analyzing query: '{user_input}' ---")
        
        if self.llm:
            try:
                structured_llm = self.llm.with_structured_output(MovieFilter)
                result = structured_llm.invoke(user_input)
                filters = result.dict()
                
                # Apply safety net for year detection
                filters = self._apply_year_safety_net(user_input, filters)
                
                # Validate genre
                if filters["genre"] not in GENRE_LIST and filters["genre"] != "Any":
                    filters["genre"] = "Any"
                
                print(f"    Filters: {filters}")
                
                vague_keywords = ['movie', 'something', 'anything', 'film', 'watch']
                if filters.get('keyword', '').lower() in vague_keywords:
                    filters['keyword'] = 'popular highly rated'  # Better default
                    print(f"    [Parser] Vague query detected, using default keyword")
                return filters
                
            except Exception as e:
                print(f"    LLM Error: {e}. Using fallback.")
        
        # Fallback
        return self._fallback_parse_query(user_input)
    
    def _apply_year_safety_net(self, user_input: str, filters: Dict) -> Dict:
        """Apply regex-based year detection as safety net."""
        # Look for specific year
        year_match = re.search(r'\b(19|20)\d{2}\b', user_input)
        if year_match:
            detected_year = int(year_match.group(0))
            if filters['year_start'] == 1900 and filters['year_end'] == 2025:
                filters['year_start'] = detected_year
                filters['year_end'] = detected_year
                print(f"    [Parser Fix] Detected year '{detected_year}'")
        
        # Look for decade references
        decade_patterns = {
            r"80s|80's|eighties": (1980, 1989),
            r"90s|90's|nineties": (1990, 1999),
            r"2000s|00s": (2000, 2009),
            r"2010s|10s": (2010, 2019),
            r"2020s|20s": (2020, 2029),
        }
        
        input_lower = user_input.lower()
        for pattern, (start, end) in decade_patterns.items():
            if re.search(pattern, input_lower):
                if filters['year_start'] == 1900 and filters['year_end'] == 2025:
                    filters['year_start'] = start
                    filters['year_end'] = end
                    print(f"    [Parser Fix] Detected decade {start}-{end}")
                break
        
        return filters
    
    def _fallback_parse_query(self, user_input: str) -> Dict:
        """Regex-based fallback for query parsing."""
        filters = {
            "genre": "Any",
            "year_start": 1900,
            "year_end": 2025,
            "keyword": user_input,
            "mood": None
        }
        
        input_lower = user_input.lower()
        
        # Extract genre
        for genre in GENRE_LIST:
            if genre.lower() in input_lower:
                filters["genre"] = genre
                break
        
        # Genre synonyms
        genre_synonyms = {
            "scary": "Horror",
            "horror": "Horror",
            "funny": "Comedy",
            "comedy": "Comedy",
            "romantic": "Romance",
            "love": "Romance",
            "action": "Action",
            "adventure": "Adventure",
            "sci-fi": "Sci-Fi",
            "science fiction": "Sci-Fi",
            "animated": "Animation",
            "cartoon": "Animation",
            "documentary": "Documentary",
            "thriller": "Thriller",
            "suspense": "Thriller",
            "war": "War",
            "western": "Western",
            "mystery": "Mystery",
            "crime": "Crime",
            "drama": "Drama",
            "fantasy": "Fantasy",
            "musical": "Musical",
        }
        
        for keyword, genre in genre_synonyms.items():
            if keyword in input_lower:
                filters["genre"] = genre
                break
        
        # Apply year safety net
        filters = self._apply_year_safety_net(user_input, filters)
        
        return filters
