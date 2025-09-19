import PyPDF2
import re
import logging
from typing import List, Dict
from io import BytesIO

logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text(self, pdf_content: bytes) -> str:
        try:
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_parts = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        cleaned_text = self._clean_text(page_text)
                        if cleaned_text:
                            text_parts.append(cleaned_text)
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num + 1}: {e}")
            
            full_text = "\n\n".join(text_parts)
            
            if not full_text.strip():
                raise ValueError("No extractable text found in PDF")
            
            logger.info(f"Extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            logger.error(f"PDF text extraction failed: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix hyphenated words across lines
        text = re.sub(r'([a-z])-\s*\n\s*([a-z])', r'\1\2', text)
        
        # Remove page numbers and simple headers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def create_chunks(self, text: str, filename: str) -> List[Dict]:
        if not text.strip():
            return []
        
        # Split into sentences for better semantic boundaries
        sentences = self._split_sentences(text)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append({
                        'chunk_id': chunk_id,
                        'filename': filename,
                        'content': current_chunk.strip(),
                        'char_count': len(current_chunk)
                    })
                    chunk_id += 1
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'filename': filename,
                'content': current_chunk.strip(),
                'char_count': len(current_chunk)
            })
        
        logger.info(f"Created {len(chunks)} chunks from {filename}")
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
