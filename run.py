import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting RAG Pipeline API on {host}:{port}")
    print(f"Documentation available at http://{host}:{port}/docs")
    
    uvicorn.run(
        "app.main:app", 
        host=host,
        port=port,
        reload=True
    )
