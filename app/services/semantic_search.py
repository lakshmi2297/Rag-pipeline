import logging
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SemanticSearch:
    def __init__(self):
        self.documents = []
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
    
    def add_document(self, document: Dict):
        self.documents.append(document)
        self._rebuild_tfidf_index()
    
    def _rebuild_tfidf_index(self):
        if self.documents:
            texts = [doc['content'] for doc in self.documents]
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                logger.debug(f"Rebuilt TF-IDF index with {len(texts)} documents")
            except ValueError as e:
                logger.warning(f"TF-IDF indexing failed: {e}")
                self.tfidf_matrix = None
    
    async def search(self, query: str, top_k: int = 5, include_keywords: bool = True) -> List[Dict]:
        if not self.documents:
            return []
        
        try:
            # Perform semantic search using embeddings
            semantic_results = await self._semantic_search(query, top_k * 2)
            
            # Perform keyword search using TF-IDF
            keyword_results = []
            if include_keywords and self.tfidf_matrix is not None:
                keyword_results = self._keyword_search(query, top_k * 2)
            
            # Combine results
            combined_results = self._combine_results(semantic_results, keyword_results)
            
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        # Placeholder for semantic search - would use actual embeddings
        results = []
        query_words = set(query.lower().split())
        
        for doc in self.documents:
            content_words = set(doc['content'].lower().split())
            overlap = len(query_words.intersection(content_words))
            score = overlap / len(query_words) if query_words else 0
            
            if score > 0:
                result = doc.copy()
                result['score'] = score
                results.append(result)
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            results = []
            for idx, score in enumerate(similarities):
                if score > 0:
                    result = self.documents[idx].copy()
                    result['score'] = float(score)
                    results.append(result)
            
            return sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _combine_results(self, semantic_results: List[Dict], keyword_results: List[Dict]) -> List[Dict]:
        # Simple combination strategy
        all_results = {}
        
        # Add semantic results
        for result in semantic_results:
            key = (result['filename'], result['chunk_id'])
            all_results[key] = result
        
        # Add keyword results, updating scores
        for result in keyword_results:
            key = (result['filename'], result['chunk_id'])
            if key in all_results:
                # Combine scores
                all_results[key]['score'] = (all_results[key]['score'] + result['score']) / 2
            else:
                all_results[key] = result
        
        # Sort by combined score
        combined = list(all_results.values())
        return sorted(combined, key=lambda x: x['score'], reverse=True)
    
    def get_stats(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "total_files": len(set(doc['filename'] for doc in self.documents)),
            "avg_chunk_length": np.mean([len(doc['content']) for doc in self.documents]) if self.documents else 0
        }
