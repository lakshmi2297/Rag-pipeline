import logging
import requests
import os
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class GenerationService:
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.base_url = "https://api.mistral.ai/v1"
        self.model = "mistral-small-latest"
        
        self.use_fallback = not self.api_key or self.api_key == "your_api_key_here"
        if self.use_fallback:
            logger.warning("Using fallback text generation")
    
    async def generate_answer(self, query: str, processed_query: str, search_results: List[Dict]) -> Tuple[str, float]:
        if self.use_fallback:
            return self._generate_fallback_answer(query, search_results)
        
        try:
            return await self._generate_mistral_answer(query, search_results)
        except Exception as e:
            logger.warning(f"Mistral generation failed: {e}")
            return self._generate_fallback_answer(query, search_results)
    
    async def _generate_mistral_answer(self, query: str, search_results: List[Dict]) -> Tuple[str, float]:
        context = self._prepare_context(search_results)
        
        prompt = f"""Use the following context to answer the question. Be concise and accurate.

Context:
{context}

Question: {query}

Answer:"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            confidence = self._calculate_confidence(search_results)
            return answer, confidence
        else:
            raise Exception(f"Mistral API error: {response.status_code}")
    
    def _generate_fallback_answer(self, query: str, search_results: List[Dict]) -> Tuple[str, float]:
        if not search_results:
            return "No relevant information found in the documents.", 0.0
        
        # Create simple template-based answer
        filenames = list(set(r['filename'] for r in search_results[:3]))
        answer = f"Based on your question '{query}', I found relevant information in {', '.join(filenames)}. "
        
        # Add content preview
        top_result = search_results[0]
        content_preview = top_result['content'][:200] + "..." if len(top_result['content']) > 200 else top_result['content']
        answer += f"Here's what I found: {content_preview}"
        
        confidence = self._calculate_confidence(search_results)
        return answer, confidence
    
    def _prepare_context(self, search_results: List[Dict]) -> str:
        context_parts = []
        for i, result in enumerate(search_results[:3], 1):
            filename = result['filename']
            content = result['content'][:300] + "..." if len(result['content']) > 300 else result['content']
            context_parts.append(f"Document {i} ({filename}): {content}")
        
        return "\n\n".join(context_parts)
    
    def _calculate_confidence(self, search_results: List[Dict]) -> float:
        if not search_results:
            return 0.0
        
        # Base confidence on average score of top results
        top_scores = [r.get('score', 0) for r in search_results[:3]]
        avg_score = sum(top_scores) / len(top_scores)
        
        # Normalize confidence score
        return min(1.0, avg_score * 1.5)
