# RAG Pipeline

A production-ready Retrieval-Augmented Generation system built with FastAPI and Mistral AI for intelligent PDF document analysis.

## Features

- 📄 **PDF Processing**: Extract and chunk text from PDF documents
- 🔍 **Semantic Search**: Hybrid search combining embeddings and TF-IDF
- 🤖 **AI Generation**: Generate contextual answers using Mistral AI
- 🌐 **FastAPI**: Modern, fast web API with automatic documentation
- 🔧 **Production Ready**: Comprehensive error handling and logging

## Quick Start

### Windows
1. Run setup: `setup.bat`
2. Start server: `start.bat`
3. Open: http://127.0.0.1:8000/docs

### Manual Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Mistral API key
python run.py
