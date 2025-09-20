import re
import logging

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self):
        # Patterns for non-search queries
        self.greeting_patterns = [
            r'^(hi|hello|hey|good morning|good afternoon)\b',
            r'^(how are you|what\'s up)\b',
            r'^(thanks|thank you|bye)\b'
        ]
        
        # Patterns that indicate search intent
        self.search_patterns = [
            r'\b(what|how|why|when|where|who|which)\b',
            r'\b(explain|describe|tell me|show me)\b',
            r'\b(find|search|document|content)\b'
        ]
    
    def detect_search_intent(self, query: str) -> bool:
        query_lower = query.lower().strip()
        
        # Check for greetings first
        for pattern in self.greeting_patterns:
            if re.search(pattern, query_lower):
                return False
        
        # Check for search indicators
        for pattern in self.search_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Default: search if query is substantial
        return len(query_lower.split()) >= 3
    
    def transform_query(self, query: str) -> str:
        # Clean query for better search performance
        transformed = query.strip()
        
        # Remove common question prefixes
        prefixes_to_remove = [
            r'^can you tell me about\s+',
            r'^please explain\s+',
            r'^what is\s+',
            r'^tell me about\s+'
        ]
        
        for prefix in prefixes_to_remove:
            transformed = re.sub(prefix, '', transformed, flags=re.IGNORECASE)
        
        return transformed.strip() if transformed.strip() else query
