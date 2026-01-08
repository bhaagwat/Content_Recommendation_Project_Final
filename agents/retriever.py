"""
Retriever - ChromaDB semantic search for movies
"""

import csv
import re
from typing import Dict, List, Set, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    MOVIES_CSV, CHROMA_DB_PATH, GENRE_TO_ID, 
    RECOMMENDATION_CONFIG, GENRE_LIST
)


class RetrieverEngine:
    """Handles semantic search over movie database using ChromaDB."""
    
    def __init__(self):
        self.collection = None
        self.movies_cache = {}  # movieId -> movie info
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB with sentence transformer embeddings."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self.collection = self.client.get_or_create_collection(
                name="movies_v3",
                embedding_function=self.ef
            )
            print(f"✓ ChromaDB initialized. Collection has {self.collection.count()} movies.")
            
        except Exception as e:
            print(f"⚠ ChromaDB initialization failed: {e}")
            self.collection = None
    
    def seed_database(self, force_reseed: bool = False):
        """Load movies from CSV into ChromaDB with popularity scores."""
        if not self.collection:
            print("❌ Cannot seed: ChromaDB not initialized")
            return
        
        if self.collection.count() > 0 and not force_reseed:
            print(f"ℹ Database already seeded with {self.collection.count()} movies")
            return
        
        if not MOVIES_CSV.exists():
            print(f"❌ Movies file not found: {MOVIES_CSV}")
            return
        
        print(f"--- [Retriever] Seeding database from {MOVIES_CSV} ---")
        
        # Increase CSV field size limit
        csv.field_size_limit(2147483647)
        
        # First pass: count movie occurrences to determine popularity
        # We'll use a simple heuristic: movies with lower IDs tend to be more popular
        # (MovieLens assigns IDs roughly by when they were added/rated)
        
        # Load all movies first
        all_movies = []
        with open(MOVIES_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_movies.append(row)
        
        print(f"    Total movies in CSV: {len(all_movies)}")
        
        # Define popular movies (first 5000 by ID, or you can use a ratings count)
        # These are more likely to be in LSTM vocabulary
        popular_movie_ids = set()
        sorted_movies = sorted(all_movies, key=lambda x: int(x.get('movieId', 999999)))
        for movie in sorted_movies[:5000]:
            popular_movie_ids.add(int(movie.get('movieId', 0)))
        
        print(f"    Marked {len(popular_movie_ids)} movies as popular")
        
        # Now seed the database
        ids, docs, metas = [], [], []
        
        for row in all_movies:
            movie_id = row.get('movieId', '')
            title = row.get('title', '')
            genres = row.get('genres', '(no genres listed)')
            
            # Extract year from title
            year = self._extract_year(title)
            
            # Get primary genre
            primary_genre = genres.split('|')[0] if genres else '(no genres listed)'
            genre_id = GENRE_TO_ID.get(primary_genre, 0)
            
            # Check if popular
            is_popular = int(movie_id) in popular_movie_ids if movie_id.isdigit() else False
            
            # Create searchable document
            doc = f"{title} {genres.replace('|', ' ')}"
            
            ids.append(movie_id)
            docs.append(doc)
            metas.append({
                "title": title,
                "genres": genres,
                "genre_id": genre_id,
                "primary_genre": primary_genre,
                "year": year,
                "movie_id": int(movie_id) if movie_id.isdigit() else 0,
                "is_popular": is_popular
            })
            
            # Batch insert
            if len(ids) >= 500:
                self.collection.add(ids=ids, documents=docs, metadatas=metas)
                print(f"    Added {self.collection.count()} movies...")
                ids, docs, metas = [], [], []
        
        # Add remaining
        if ids:
            self.collection.add(ids=ids, documents=docs, metadatas=metas)
        
        print(f"✓ Database seeded with {self.collection.count()} movies")
        
        ids, docs, metas = [], [], []
        
        with open(MOVIES_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                movie_id = row.get('movieId', '')
                title = row.get('title', '')
                genres = row.get('genres', '(no genres listed)')
                
                # Extract year from title
                year = self._extract_year(title)
                
                # Get primary genre
                primary_genre = genres.split('|')[0] if genres else '(no genres listed)'
                genre_id = GENRE_TO_ID.get(primary_genre, 0)
                
                # Create searchable document
                # Combine title and genres for semantic search
                doc = f"{title} {genres.replace('|', ' ')}"
                
                ids.append(movie_id)
                docs.append(doc)
                metas.append({
                    "title": title,
                    "genres": genres,
                    "genre_id": genre_id,
                    "primary_genre": primary_genre,
                    "year": year,
                    "movie_id": int(movie_id) if movie_id.isdigit() else 0
                })
                
                # Batch insert
                if len(ids) >= 500:
                    self.collection.add(ids=ids, documents=docs, metadatas=metas)
                    print(f"    Added {self.collection.count()} movies...")
                    ids, docs, metas = [], [], []
        
        # Add remaining
        if ids:
            self.collection.add(ids=ids, documents=docs, metadatas=metas)
        
        print(f"✓ Database seeded with {self.collection.count()} movies")
    
    def _extract_year(self, title: str) -> int:
        """Extract year from movie title like 'Toy Story (1995)'."""
        match = re.search(r'\((\d{4})\)', title)
        if match:
            return int(match.group(1))
        return 2000  # Default
    
    def search(
        self, 
        filters: Dict, 
        k: int = None,
        exclude_ids: Set[int] = None,
        popular_only: bool = False
    ) -> List[Dict]:
        """
        Search for movies matching filters.
        
        Args:
            filters: {genre, year_start, year_end, keyword, mood}
            k: Number of results (default from config)
            exclude_ids: Movie IDs to exclude (already watched)
            popular_only: If True, only return popular movies
        
        Returns:
            List of movie candidates with embeddings
        """
        if not self.collection:
            print("❌ Cannot search: ChromaDB not initialized")
            return []
        
        k = k or RECOMMENDATION_CONFIG["retrieval_candidates"]
        exclude_ids = exclude_ids or set()
        
        print(f"--- [Retriever] Searching for: '{filters.get('keyword', '')}' ---")
        
        # Build where clause for filtering
        where_clause = self._build_where_clause(filters, popular_only)
        
        # Build query text
        query_text = filters.get("keyword", "")
        if filters.get("mood"):
            query_text += f" {filters['mood']}"
        
        try:
            # Query with filters
            if where_clause:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=k * 2,
                    include=['documents', 'metadatas', 'embeddings'],
                    where=where_clause
                )
            else:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=k * 2,
                    include=['documents', 'metadatas', 'embeddings']
                )
            
            # Process results
            candidates = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    movie_id = int(results['ids'][0][i])
                    
                    if movie_id in exclude_ids:
                        continue
                    
                    candidates.append({
                        "id": movie_id,
                        "title": results['metadatas'][0][i]['title'],
                        "metadata": results['metadatas'][0][i],
                        "embedding": results['embeddings'][0][i] if results['embeddings'] else None
                    })
                    
                    if len(candidates) >= k:
                        break
            
            print(f"    Found {len(candidates)} candidates")
            return candidates
            
        except Exception as e:
            print(f"    [Error] Search failed: {e}")
            return []
    
    def _build_where_clause(self, filters: Dict, popular_only: bool = False) -> Optional[Dict]:
        """Build ChromaDB where clause from filters."""
        conditions = []
        
        # Year filter
        year_start = filters.get('year_start', 1900)
        year_end = filters.get('year_end', 2025)
        
        if year_start != 1900 or year_end != 2025:
            conditions.append({"year": {"$gte": year_start}})
            conditions.append({"year": {"$lte": year_end}})
        
        # Genre filter
        genre = filters.get('genre', 'Any')
        if genre and genre != 'Any' and genre in GENRE_LIST:
            conditions.append({"primary_genre": {"$eq": genre}})
        
        # Popularity filter
        if popular_only:
            conditions.append({"is_popular": {"$eq": True}})
        
        # Build final clause
        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
    
    def search_by_preferences(
        self, 
        preferences: Dict, 
        k: int = 20,
        exclude_ids: Set[int] = None,
        popular_only: bool = True
    ) -> List[Dict]:
        """
        Search for movies matching user preferences (for onboarding).
        
        Args:
            preferences: {genres, actors, directors, keywords}
            k: Number of results
            exclude_ids: Movie IDs to exclude
            popular_only: If True, only return popular movies (in LSTM vocab)
        """
        if not self.collection:
            return []
        
        exclude_ids = exclude_ids or set()
        
        # Build query from preferences
        query_parts = []
        
        if preferences.get("genres"):
            query_parts.extend(preferences["genres"])
        
        if preferences.get("keywords"):
            query_parts.extend(preferences["keywords"])
        
        # Add actor/director names as keywords (semantic search might help)
        if preferences.get("directors"):
            query_parts.extend(preferences["directors"])
        
        if preferences.get("actors"):
            query_parts.extend(preferences["actors"])
        
        query_text = " ".join(query_parts) if query_parts else "popular classic movie"
        
        print(f"--- [Retriever] Preference search: '{query_text}' ---")
        print(f"    Popular only: {popular_only}")
        
        try:
            # Build where clause
            where_clause = None
            if popular_only:
                where_clause = {"is_popular": {"$eq": True}}
            
            # Query ChromaDB
            if where_clause:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=k * 3,  # Get extra to account for exclusions
                    include=['documents', 'metadatas', 'embeddings'],
                    where=where_clause
                )
            else:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=k * 3,
                    include=['documents', 'metadatas', 'embeddings']
                )
            
            candidates = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    movie_id = int(results['ids'][0][i])
                    
                    if movie_id in exclude_ids:
                        continue
                    
                    candidates.append({
                        "id": movie_id,
                        "title": results['metadatas'][0][i]['title'],
                        "metadata": results['metadatas'][0][i],
                        "embedding": results['embeddings'][0][i] if results['embeddings'] else None
                    })
                    
                    if len(candidates) >= k:
                        break
            
            print(f"    Found {len(candidates)} candidates")
            return candidates
            
        except Exception as e:
            print(f"    [Error] Preference search failed: {e}")
            return []
    
    def get_movie_by_id(self, movie_id: int) -> Optional[Dict]:
        """Get a specific movie by ID."""
        if not self.collection:
            return None
        
        try:
            results = self.collection.get(
                ids=[str(movie_id)],
                include=['metadatas', 'embeddings']
            )
            
            if results['ids']:
                return {
                    "id": movie_id,
                    "title": results['metadatas'][0]['title'],
                    "metadata": results['metadatas'][0],
                    "embedding": results['embeddings'][0] if results['embeddings'] else None
                }
        except:
            pass
        
        return None
    
    def get_movies_by_ids(self, movie_ids: List[int]) -> List[Dict]:
        """Get multiple movies by their IDs."""
        movies = []
        for mid in movie_ids:
            movie = self.get_movie_by_id(mid)
            if movie:
                movies.append(movie)
        return movies
