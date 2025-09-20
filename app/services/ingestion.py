import logging
from typing import List, Dict
from ..utils.pdf_extractor import PDFExtractor
from ..utils.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.embedding_service = EmbeddingService()
        self.processed_documents = []
    
    async def process_document(self, content: bytes, filename: str) -> List[Dict]:
        try:
            logger.info(f"Processing document: {filename}")
            
            # Extract text from PDF
            text = self.pdf_extractor.extract_text(content)
            
            # Create semantic chunks
            chunks = self.pdf_extractor.create_chunks(text, filename)
            
            # Generate embeddings for each chunk
            processed_chunks = []
            for chunk in chunks:
                embedding = await self.embedding_service.get_embedding(chunk['content'])
                chunk['embedding'] = embedding
                processed_chunks.append(chunk)
            
            # Store processed chunks
            self.processed_documents.extend(processed_chunks)
            
            logger.info(f"Successfully processed {filename}: {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Document processing failed for {filename}: {e}")
            raise
