# RAG PDF Bot

This project is a simple Retrieval-Augmented Generation (RAG) application that allows you to query a PDF and receive answers grounded in the PDF's content. It uses embeddings to chunk and store the PDF data in a vector database, then queries it using a local LLM.

## Features
- Upload and index PDF documents.
- Query natural language questions against the PDF content.
- Retrieve top relevant chunks from the database.
- Generate answers using a local LLM with the retrieved context.
- Basic web frontend for entering questions and viewing answers with supporting chunks.

## Project Structure
- `query_data.py`: Command-line interface for querying the PDF.
- `app.py`: Flask backend to serve queries via a web API.
- `index.html`: Frontend for entering questions and displaying results.
- `CHROMA_PATH`: Persistent vector database storage.

## Requirements
- Python 3.9+
- Flask
- LangChain
- Ollama
- ChromaDB
- PyPDF or similar PDF parsing library

Install dependencies with:
```bash
pip install -r requirements.txt

```

## Running Project
- CLI query: python query_data.py "What is a graph?"
- Start web app: python app.py
- Open web app (paste in browser): http://127.0.0.1:5000

## Credits
This project is based on the tutorial:

Python RAG Tutorial (with Local LLMs): AI For Your PDFs by pixegami 

Special thanks to pixegami for the educational content.

