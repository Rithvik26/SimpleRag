"""
SimpleRAG - A Retrieval-Augmented Generation System for document Q&A (Web App Version)

This system allows users to:
1. Index documents (PDF, TXT, DOCX) into a Qdrant vector database using Gemini embeddings
2. Query the documents using natural language
3. Receive contextually relevant answers powered by Claude LLMs

This modified version is designed to work with the Flask web application.
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Document processing
import PyPDF2
import docx
import tiktoken
from bs4 import BeautifulSoup

# Vector DB and embeddings
import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import requests

# LLM integration
import anthropic

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('SimpleRAG')

# Configuration - for web app, we'll use environment variables when available
DEFAULT_CONFIG = {
    "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
    "claude_api_key": os.environ.get("CLAUDE_API_KEY", ""),
    "qdrant_url": os.environ.get("QDRANT_URL", "https://3cbcacc0-1fe5-42a1-8be0-81515a21771b.us-west-2-0.aws.cloud.qdrant.io"),
    "qdrant_api_key": os.environ.get("QDRANT_API_KEY", ""),
    "collection_name": os.environ.get("QDRANT_COLLECTION", "simple_rag_docs"),
    "embedding_dimension": 768,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 5,
    "preferred_llm": "claude"
}

# Default config path, but for web app we might store in temp directory
CONFIG_PATH = os.environ.get("CONFIG_PATH", os.path.expanduser("~/.simplerag/config.json"))

def ensure_config_exists():
    """Make sure config file exists, create with defaults if not."""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        logger.info(f"Created default configuration at {CONFIG_PATH}")
        logger.info("Please edit this file to add your API keys.")
        return False
    return True

def load_config():
    """Load configuration from disk."""
    ensure_config_exists()
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # Override with environment variables if set
    if os.environ.get("GEMINI_API_KEY"):
        config["gemini_api_key"] = os.environ.get("GEMINI_API_KEY")
    if os.environ.get("CLAUDE_API_KEY"):
        config["claude_api_key"] = os.environ.get("CLAUDE_API_KEY")
    if os.environ.get("QDRANT_API_KEY"):
        config["qdrant_api_key"] = os.environ.get("QDRANT_API_KEY")
    if os.environ.get("QDRANT_URL"):
        config["qdrant_url"] = os.environ.get("QDRANT_URL")
    
    # Debug output for web app
    print(f"Config loaded: {config}")
    
    return config

class DocumentProcessor:
    """Handles document parsing, chunking, and text extraction."""
    
    def __init__(self, config):
        self.chunk_size = config["chunk_size"]
        self.chunk_overlap = config["chunk_overlap"]
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif extension == '.txt':
            return self._extract_from_txt(file_path)
        elif extension == '.docx':
            return self._extract_from_docx(file_path)
        elif extension == '.html' or extension == '.htm':
            return self._extract_from_html(file_path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            return file.read()
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def _extract_from_html(self, file_path: str) -> str:
        """Extract text from HTML file."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
    
    def chunk_text(self, text: str, metadata: dict) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks for embedding."""
        if not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        # Split text into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, add current chunk to list
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_text"] = current_chunk[:100] + "..."  # Preview
                chunks.append({
                    "text": current_chunk,
                    "metadata": chunk_metadata
                })
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_word_count = min(len(words), self.chunk_overlap // 10)
                current_chunk = " ".join(words[-overlap_word_count:]) + " " + sentence
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_text"] = current_chunk[:100] + "..."  # Preview
            chunks.append({
                "text": current_chunk,
                "metadata": chunk_metadata
            })
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks

class EmbeddingService:
    """Handles creation of embeddings using Gemini API."""
    
    def __init__(self, config):
        self.api_key = config["gemini_api_key"]
        self.embedding_dimension = config["embedding_dimension"]
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text using Gemini API."""
        url = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"
        
        headers = {
            "Content-Type": "application/json",
        }
        
        params = {
            "key": self.api_key
        }
        
        data = {
            "model": "models/embedding-001",
            "content": {"parts": [{"text": text}]}
        }
        
        try:
            response = requests.post(url, headers=headers, params=params, json=data)
            response.raise_for_status()
            result = response.json()
            
            # Extract embedding vector from response
            embedding = result.get("embedding", {}).get("values", [])
            
            if not embedding:
                raise ValueError("No embedding values returned from API")
                
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            logger.error(f"API Response: {response.text if 'response' in locals() else 'No response'}")
            raise

class VectorDBService:
    """Handles interactions with Qdrant vector database."""
    
    def __init__(self, config):
        self.qdrant_url = config["qdrant_url"]
        self.qdrant_api_key = config["qdrant_api_key"]
        self.collection_name = config["collection_name"]
        self.embedding_dimension = config["embedding_dimension"]
        
        self.client = qdrant_client.QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )
    
    def ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            return True
        return False
    
    def insert_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Insert documents with their embeddings into Qdrant."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
        
        self.ensure_collection_exists()
        
        # Prepare points for insertion
        points = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            points.append(
                models.PointStruct(
                    id=i + int(time.time() * 1000),  # Generate a unique ID
                    vector=embedding,
                    payload={
                        "text": doc["text"],
                        "metadata": doc["metadata"]
                    }
                )
            )
        
        # Insert in batches to avoid overloading the API
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            logger.info(f"Inserted batch of {len(batch)} documents")
        
        logger.info(f"Successfully inserted {len(documents)} documents into Qdrant")
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5, filter_condition=None) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        self.ensure_collection_exists()
        
        # Prepare filter - useful for filtering by session_id in web app
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_embedding,
            "limit": top_k
        }
        
        if filter_condition:
            search_params["query_filter"] = filter_condition
        
        # For newer versions of qdrant-client, use query_points instead of search
        try:
            # First try the newer API
            query_params = search_params.copy()
            query_params["query_vector"] = search_params.get("query_vector")
            if "query_vector" in search_params:
                del query_params["query_vector"]
            
            search_result = self.client.query_points(**query_params)
            
            results = []
            for scored_point in search_result:
                results.append({
                    "text": scored_point.payload["text"],
                    "metadata": scored_point.payload["metadata"],
                    "score": scored_point.score
                })
        except (AttributeError, TypeError):
            # Fall back to older API for compatibility
            search_result = self.client.search(**search_params)
            
            results = []
            for scored_point in search_result:
                results.append({
                    "text": scored_point.payload["text"],
                    "metadata": scored_point.payload["metadata"],
                    "score": scored_point.score
                })
        
        return results

class LLMService:
    """Handles interactions with LLM APIs (Claude)."""
    
    def __init__(self, config):
        self.preferred_llm = config["preferred_llm"]
        self.claude_api_key = config.get("claude_api_key", "")
        
        if self.preferred_llm == "claude" and self.claude_api_key:
            try:
                self.claude_client = anthropic.Anthropic(api_key=self.claude_api_key)
            except TypeError:
                # Fall back to a method that doesn't use proxies
                import httpx
                http_client = httpx.Client()
                self.claude_client = anthropic.Anthropic(
                    api_key=self.claude_api_key,
                    http_client=http_client
                )
    
    def generate_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Generate an answer based on the query and retrieved contexts."""
        # Prepare context text from retrieved documents
        context_text = "\n\n---\n\n".join([
            f"Document: {ctx['metadata'].get('filename', 'Unknown')}\n{ctx['text']}"
            for ctx in contexts
        ])
        
        prompt = f"""
You are a helpful assistant that answers questions based on the provided document context.

CONTEXT:
{context_text}

USER QUESTION:
{query}

Please answer the question based only on the provided context. If the answer cannot be determined from the context, please state that clearly. Include relevant citations to the documents when possible.

ANSWER:
"""
        
        if self.preferred_llm == "claude" and hasattr(self, "claude_client"):
            return self._generate_with_claude(prompt)
        elif self.preferred_llm == "raw" or not hasattr(self, "claude_client"):
            # Raw mode or no LLM configured - just return formatted contexts
            return self._generate_raw_response(query, contexts)
        else:
            raise ValueError(f"LLM {self.preferred_llm} not properly configured")
    
    def _generate_with_claude(self, prompt: str) -> str:
        """Generate answer using Claude API."""
        try:
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating response with Claude: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _generate_raw_response(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Generate a formatted response with just the relevant contexts (no LLM)."""
        response = [f"Results for query: '{query}'\n"]
        
        for i, ctx in enumerate(contexts):
            response.append(f"--- Result {i+1} (Relevance: {ctx['score']:.4f}) ---")
            response.append(f"Document: {ctx['metadata'].get('filename', 'Unknown')}")
            if 'page' in ctx['metadata']:
                response.append(f"Page: {ctx['metadata']['page']}")
            response.append("")  # Empty line
            response.append(ctx['text'])
            response.append("")  # Empty line
        
        return "\n".join(response)

class SimpleRAG:
    """Main class that orchestrates the RAG system."""
    
    def __init__(self):
        self.config = load_config()
        self.document_processor = DocumentProcessor(self.config)
        self.embedding_service = EmbeddingService(self.config)
        self.vector_db_service = VectorDBService(self.config)
        self.llm_service = LLMService(self.config)
    
    def index_document(self, file_path: str, session_id=None) -> bool:
        """Process and index a document into the vector database."""
        try:
            # Extract text from document
            logger.info(f"Extracting text from {file_path}")
            text = self.document_processor.extract_text_from_file(file_path)
            
            # Create metadata
            filename = os.path.basename(file_path)
            metadata = {
                "filename": filename,
                "path": file_path,
                "created_at": time.time(),
                "file_type": os.path.splitext(filename)[1][1:].lower()
            }
            
            # Add session ID if provided (for web app)
            if session_id:
                metadata["session_id"] = session_id
            
            # Chunk the document
            chunks = self.document_processor.chunk_text(text, metadata)
            
            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                return False
            
            # Generate embeddings for each chunk
            embeddings = []
            for chunk in chunks:
                logger.info(f"Generating embedding for chunk from {filename}")
                embedding = self.embedding_service.get_embedding(chunk["text"])
                embeddings.append(embedding)
            
            # Store in vector database
            logger.info(f"Storing {len(chunks)} chunks in vector database")
            self.vector_db_service.insert_documents(chunks, embeddings)
            
            logger.info(f"Successfully indexed document: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing document {file_path}: {str(e)}")
            return False
    
    def query(self, question: str, session_id=None) -> str:
        """Query the indexed documents and return an answer."""
        try:
            # Generate embedding for the query
            logger.info(f"Generating embedding for query: {question}")
            query_embedding = self.embedding_service.get_embedding(question)
            
            # Prepare filter if session_id provided
            filter_condition = None
            if session_id:
                filter_condition = {
                    "must": [
                        {
                            "key": "metadata.session_id",
                            "match": {"value": session_id}
                        }
                    ]
                }
            
            # Retrieve similar contexts
            logger.info(f"Searching for relevant contexts")
            contexts = self.vector_db_service.search_similar(
                query_embedding,
                top_k=self.config["top_k"],
                filter_condition=filter_condition
            )
            
            if not contexts:
                logger.warning("No relevant contexts found")
                return "I couldn't find any relevant information to answer your question."
            
            # Generate answer using LLM
            logger.info(f"Generating answer using {self.config['preferred_llm']}")
            answer = self.llm_service.generate_answer(question, contexts)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Error processing your query: {str(e)}"

def main():
    """Main entry point for the command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SimpleRAG - Document Q&A System")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configure the system")
    config_parser.add_argument("--gemini-key", help="Set Gemini API key")
    config_parser.add_argument("--claude-key", help="Set Claude API key")
    config_parser.add_argument("--qdrant-key", help="Set Qdrant API key")
    config_parser.add_argument("--preferred-llm", choices=["claude", "raw"], help="Set preferred LLM")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index a document")
    index_parser.add_argument("file_path", help="Path to the document to index")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the indexed documents")
    query_parser.add_argument("question", help="The question to ask")
    
    args = parser.parse_args()
    
    if args.command == "config":
        ensure_config_exists()
        config = load_config()
        
        if args.gemini_key:
            config["gemini_api_key"] = args.gemini_key
        if args.claude_key:
            config["claude_api_key"] = args.claude_key
        if args.qdrant_key:
            config["qdrant_api_key"] = args.qdrant_key
        if args.preferred_llm:
            config["preferred_llm"] = args.preferred_llm
        
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration updated successfully!")
    
    elif args.command == "index":
        rag = SimpleRAG()
        success = rag.index_document(args.file_path)
        if success:
            print(f"Document indexed successfully: {args.file_path}")
        else:
            print(f"Failed to index document: {args.file_path}")
    
    elif args.command == "query":
        rag = SimpleRAG()
        answer = rag.query(args.question)
        print("\nAnswer:")
        print("-------")
        print(answer)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()