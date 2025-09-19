import logging
import requests
import os
import hashlib
import numpy as np
from typing import List

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.base_url = "https://api.mistral.ai/v1"
        self.model = "mistral-embed"
        
        if not self.api_key or self.api_key == "your_api_key_here":
            logger.warning("Mistral API key not configured - using fallback embeddings")
            self.use_fallback = True
        else:
            logger.info("Mistral AI embedding service initialized")
            self.use_fallback = False
    
    async def get_embedding(self, text: str) -> List[float]:
        if self.use_fallback:
            return self._generate_fallback_embedding(text)
        
        try:
            return await self._get_mistral_embedding(text)
        except Exception as e:
            logger.warning(f"Mistral API failed, using fallback: {e}")
            return self._generate_fallback_embedding(text)
    
    async def _get_mistral_embedding(self, text: str) -> List[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": text
        }
        
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["data"][0]["embedding"]
        else:
            raise Exception(f"Mistral API error: {response.status_code}")
    
    def _generate_fallback_embedding(self, text: str) -> List[float]:
        # Hash-based embedding for development/testing
        text_hash = hashlib.md5(text.encode()).hexdigest()
        hash_bytes = bytes.fromhex(text_hash)
        
        # Convert to numbers and extend to 1024 dimensions
        numbers = list(hash_bytes) * 64  # 16 * 64 = 1024
        numbers = numbers[:1024]
        
        # Normalize to unit vector
        arr = np.array(numbers, dtype=float)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        
        return arr.tolist()
