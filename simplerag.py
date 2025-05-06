"""
SimpleRAG - A Retrieval-Augmented Generation System for document Q&A

This system allows users to:
1. Index documents (PDF, TXT, DOCX, HTML) into a Qdrant vector database using Gemini embeddings
2. Query the documents using natural language
3. Receive contextually relevant answers powered by Claude LLMs

Enhanced with rate limiting, caching, and progress tracking.
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

# Import extensions
from extensions import RateLimiter, EmbeddingCache, rate_limited, cached_embedding, ProgressTracker

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
    "preferred_llm": "claude",
    "rate_limit": 60,  # Max API calls per minute
    "enable_cache": True,  # Enable embedding cache
    "cache_dir": None  # Default cache directory
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
    
    # Add default values for new config options if not present
    if "rate_limit" not in config:
        config["rate_limit"] = DEFAULT_CONFIG["rate_limit"]
    if "enable_cache" not in config:
        config["enable_cache"] = DEFAULT_CONFIG["enable_cache"]
    if "cache_dir" not in config:
        config["cache_dir"] = DEFAULT_CONFIG["cache_dir"]
    
    # Debug output for web app
    print(f"Config loaded: {config}")
    
    return config

class DocumentProcessor:
    """Handles document parsing, chunking, and text extraction."""
    
    def __init__(self, config):
        self.chunk_size = config["chunk_size"]
        self.chunk_overlap = config["chunk_overlap"]
    
    def extract_text_from_file(self, file_path: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Extract text from various file formats."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if progress_tracker:
            progress_tracker.update(0, 100, status="extracting", 
                                   message=f"Extracting text from {path.name}",
                                   current_file=path.name)
        
        if extension == '.pdf':
            return self._extract_from_pdf(file_path, progress_tracker)
        elif extension == '.txt':
            return self._extract_from_txt(file_path, progress_tracker)
        elif extension == '.docx':
            return self._extract_from_docx(file_path, progress_tracker)
        elif extension == '.html' or extension == '.htm':
            return self._extract_from_html(file_path, progress_tracker)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _extract_from_pdf(self, file_path: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Extract text from PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            
            for i, page in enumerate(reader.pages):
                text += page.extract_text() + "\n"
                
                if progress_tracker and total_pages > 0:
                    progress_tracker.update(i + 1, total_pages, 
                                           message=f"Extracting page {i + 1} of {total_pages}")
        return text
    
    def _extract_from_txt(self, file_path: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Extract text from TXT file."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            text = file.read()
            if progress_tracker:
                progress_tracker.update(1, 1, status="complete", 
                                       message="Text file extraction complete")
            return text
    
    def _extract_from_docx(self, file_path: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Extract text from DOCX file."""
        doc = docx.Document(file_path)
        paragraphs = doc.paragraphs
        total_paragraphs = len(paragraphs)
        
        text_parts = []
        for i, paragraph in enumerate(paragraphs):
            text_parts.append(paragraph.text)
            
            if progress_tracker and total_paragraphs > 0:
                progress_tracker.update(i + 1, total_paragraphs, 
                                       message=f"Extracting paragraph {i + 1} of {total_paragraphs}")
        
        return "\n".join(text_parts)
    
    def _extract_from_html(self, file_path: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Extract text from HTML file."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            html_content = file.read()
            
            if progress_tracker:
                progress_tracker.update(1, 2, message="Parsing HTML content")
                
            soup = BeautifulSoup(html_content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
                
            if progress_tracker:
                progress_tracker.update(2, 2, message="Extracting text content")
                
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text
    
    def chunk_text(self, text: str, metadata: dict, progress_tracker: Optional[ProgressTracker] = None) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks for embedding."""
        if not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        if progress_tracker:
            progress_tracker.update(0, 100, status="chunking", 
                                   message="Splitting text into chunks")
        
        # Split text into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        total_sentences = len(sentences)
        
        chunks = []
        current_chunk = ""
        
        for i, sentence in enumerate(sentences):
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
                    
            # Update progress
            if progress_tracker and total_sentences > 0:
                progress_tracker.update(i + 1, total_sentences, 
                                       message=f"Processing sentence {i + 1} of {total_sentences}")
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_text"] = current_chunk[:100] + "..."  # Preview
            chunks.append({
                "text": current_chunk,
                "metadata": chunk_metadata
            })
        
        if progress_tracker:
            progress_tracker.update(total_sentences, total_sentences, status="chunking_complete",
                                   message=f"Created {len(chunks)} chunks from document")
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks

class EmbeddingService:
    """Handles creation of embeddings using Gemini API."""
    
    def __init__(self, config):
        self.api_key = config["gemini_api_key"]
        self.embedding_dimension = config["embedding_dimension"]
        
        # Set up rate limiter
        self.rate_limiter = RateLimiter(calls_per_minute=config.get("rate_limit", 60))
        
        # Set up embedding cache if enabled
        self.enable_cache = config.get("enable_cache", True)
        if self.enable_cache:
            self.cache = EmbeddingCache(cache_dir=config.get("cache_dir"))
    
    @rate_limited(RateLimiter(calls_per_minute=60))  # Default rate limiter as fallback
    def _get_embedding_from_api(self, text: str) -> List[float]:
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
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using cache if enabled."""
        # Try to get from cache first if enabled
        if self.enable_cache and hasattr(self, 'cache'):
            cached_embedding = self.cache.get(text)
            if cached_embedding is not None:
                logger.info("Using cached embedding")
                return cached_embedding
        
        # Apply rate limiting via the decorator on _get_embedding_from_api
        embedding = self._get_embedding_from_api(text)
        
        # Store in cache if enabled
        if self.enable_cache and hasattr(self, 'cache'):
            self.cache.set(text, embedding)
        
        return embedding
    
    def get_embeddings_batch(self, texts: List[str], progress_tracker: Optional[ProgressTracker] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts with progress tracking."""
        embeddings = []
        total_texts = len(texts)
        
        if progress_tracker:
            progress_tracker.update(0, total_texts, status="embedding", 
                                   message="Generating embeddings")
        
        for i, text in enumerate(texts):
            try:
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
                
                if progress_tracker:
                    progress_tracker.update(i + 1, total_texts, 
                                           message=f"Generated embedding {i + 1} of {total_texts}")
            except Exception as e:
                logger.error(f"Error generating embedding for chunk {i}: {str(e)}")
                # Add a placeholder embedding in case of error
                embeddings.append([0.0] * self.embedding_dimension)
                
                if progress_tracker:
                    progress_tracker.update(i + 1, total_texts, 
                                           message=f"Error with embedding {i + 1}: {str(e)}")
        
        return embeddings

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
    
    def insert_documents(self, documents: List[Dict[str, Any]], embeddings: List[List[float]], 
                         progress_tracker: Optional[ProgressTracker] = None):
        """Insert documents with their embeddings into Qdrant."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
        
        if progress_tracker:
            progress_tracker.update(0, len(documents), status="storing", 
                                   message="Storing documents in vector database")
        
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
            
            if progress_tracker:
                # Calculate overall progress
                progress = min(i + batch_size, len(points))
                progress_tracker.update(progress, len(points), 
                                       message=f"Stored {progress} of {len(points)} chunks")
            
            logger.info(f"Inserted batch of {len(batch)} documents")
        
        if progress_tracker:
            progress_tracker.update(len(documents), len(documents), status="complete", 
                                   message=f"Successfully stored {len(documents)} chunks")
        
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
            query_params["vector"] = search_params.get("query_vector")
            if "query_vector" in search_params:
                del query_params["query_vector"]
            
            search_result = self.client.search(**query_params)
            
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
    
    def generate_answer(self, query: str, contexts: List[Dict[str, Any]], progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Generate an answer based on the query and retrieved contexts."""
        if progress_tracker:
            progress_tracker.update(70, 100, status="generating", 
                                  message="Generating answer with LLM")
            
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
            return self._generate_with_claude(prompt, progress_tracker)
        elif self.preferred_llm == "raw" or not hasattr(self, "claude_client"):
            # Raw mode or no LLM configured - just return formatted contexts
            if progress_tracker:
                progress_tracker.update(90, 100, status="formatting", 
                                      message="Preparing raw document results")
            return self._generate_raw_response(query, contexts)
        else:
            raise ValueError(f"LLM {self.preferred_llm} not properly configured")
    
    def _generate_with_claude(self, prompt: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
        """Generate answer using Claude API."""
        try:
            if progress_tracker:
                progress_tracker.update(80, 100, status="querying", 
                                      message="Querying Claude API")
                
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            if progress_tracker:
                progress_tracker.update(95, 100, status="formatting", 
                                      message="Response received, formatting answer")
                
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
            # Create progress tracker if session_id provided
            progress_tracker = None
            if session_id:
                progress_tracker = ProgressTracker(session_id, "index_document")
                progress_tracker.update(0, 100, status="starting", 
                                      message="Starting document indexing process")
            
            # Extract text from document
            logger.info(f"Extracting text from {file_path}")
            text = self.document_processor.extract_text_from_file(file_path, progress_tracker)
            
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
            chunks = self.document_processor.chunk_text(text, metadata, progress_tracker)
            
            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                if progress_tracker:
                    progress_tracker.update(100, 100, status="error", 
                                          message="No content could be extracted from document")
                return False
            
            # Generate embeddings for each chunk
            if progress_tracker:
                progress_tracker.update(0, len(chunks), status="embedding", 
                                      message="Generating embeddings")
            
            # Extract texts from chunks for batch embedding
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_service.get_embeddings_batch(chunk_texts, progress_tracker)
            
            # Store in vector database
            logger.info(f"Storing {len(chunks)} chunks in vector database")
            self.vector_db_service.insert_documents(chunks, embeddings, progress_tracker)
            
            if progress_tracker:
                progress_tracker.update(100, 100, status="complete", 
                                      message=f"Successfully indexed document: {filename}")
            
            logger.info(f"Successfully indexed document: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing document {file_path}: {str(e)}")
            if progress_tracker:
                progress_tracker.update(0, 100, status="error", 
                                      message=f"Error: {str(e)}")
            return False
    
    def query(self, question: str, session_id=None) -> str:
        """Query indexed documents and return answer."""
        # Create progress tracker if session_id provided
        progress_tracker = None
        if session_id:
            progress_tracker = ProgressTracker(session_id, "query")
            progress_tracker.update(0, 100, status="starting", 
                                  message="Starting query processing")
        
        try:
            # Generate embedding for the query
            logger.info(f"Generating embedding for query: {question}")
            if progress_tracker:
                progress_tracker.update(10, 100, status="embedding", 
                                      message="Generating query embedding")
                
            query_embedding = self.embedding_service.get_embedding(question)
            
            # Prepare filter if needed
            filter_condition = None
            if session_id:
                # Optional: Filter by session_id to only query documents from this session
                # Uncomment if you want this behavior
                # filter_condition = models.Filter(
                #     must=[
                #         models.FieldCondition(
                #             key="metadata.session_id",
                #             match=models.MatchValue(value=session_id)
                #         )
                #     ]
                # )
                pass
            
            # Retrieve similar contexts
            logger.info(f"Searching for relevant contexts")
            if progress_tracker:
                progress_tracker.update(30, 100, status="searching", 
                                      message="Searching for relevant documents")
                
            contexts = self.vector_db_service.search_similar(
                query_embedding,
                top_k=self.config["top_k"],
                filter_condition=filter_condition
            )
            
            if not contexts:
                logger.warning("No relevant contexts found")
                if progress_tracker:
                    progress_tracker.update(100, 100, status="complete", 
                                          message="No relevant information found")
                return "I couldn't find any relevant information to answer your question."
            
            # If LLM service is available, use it to generate an answer
            if hasattr(self, "llm_service") and self.llm_service:
                logger.info(f"Generating answer using LLM")
                if progress_tracker:
                    progress_tracker.update(50, 100, status="generating", 
                                          message="Generating answer")
                
                answer = self.llm_service.generate_answer(question, contexts, progress_tracker)
                
                if progress_tracker:
                    progress_tracker.update(100, 100, status="complete", 
                                          message="Answer generated successfully")
                
                return answer
            else:
                # If no LLM, just return the relevant chunks
                logger.info(f"No LLM configured, returning raw chunks")
                if progress_tracker:
                    progress_tracker.update(100, 100, status="complete", 
                                          message="Results collected (raw mode)")
                
                results = []
                for i, ctx in enumerate(contexts):
                    results.append(f"--- Result {i+1} (Score: {ctx['score']:.2f}) ---\n")
                    results.append(f"Source: {ctx['metadata'].get('filename', 'Unknown')}\n")
                    results.append(f"{ctx['text']}\n\n")
                return "\n".join(results)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            if progress_tracker:
                progress_tracker.update(100, 100, status="error", 
                                      message=f"Error: {str(e)}")
            return f"Error processing your query: {str(e)}"

# Command line interface
def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SimpleRAG - A Retrieval-Augmented Generation system for document Q&A")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configure API keys and settings")
    config_parser.add_argument("--gemini-key", help="Set Gemini API key")
    config_parser.add_argument("--claude-key", help="Set Claude API key")
    config_parser.add_argument("--qdrant-key", help="Set Qdrant API key")
    config_parser.add_argument("--qdrant-url", help="Set Qdrant URL")
    config_parser.add_argument("--preferred-llm", choices=["claude", "raw"], help="Set preferred LLM")
    config_parser.add_argument("--chunk-size", type=int, help="Set chunk size")
    config_parser.add_argument("--chunk-overlap", type=int, help="Set chunk overlap")
    config_parser.add_argument("--top-k", type=int, help="Set number of results to retrieve")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index a document")
    index_parser.add_argument("file_path", help="Path to document to index")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query indexed documents")
    query_parser.add_argument("question", help="Question to ask")
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == "config":
        # Update configuration
        config = load_config()
        modified = False
        
        if args.gemini_key:
            config["gemini_api_key"] = args.gemini_key
            modified = True
        
        if args.claude_key:
            config["claude_api_key"] = args.claude_key
            modified = True
        
        if args.qdrant_key:
            config["qdrant_api_key"] = args.qdrant_key
            modified = True
        
        if args.qdrant_url:
            config["qdrant_url"] = args.qdrant_url
            modified = True
        
        if args.preferred_llm:
            config["preferred_llm"] = args.preferred_llm
            modified = True
        
        if args.chunk_size:
            config["chunk_size"] = args.chunk_size
            modified = True
        
        if args.chunk_overlap:
            config["chunk_overlap"] = args.chunk_overlap
            modified = True
        
        if args.top_k:
            config["top_k"] = args.top_k
            modified = True
        
        if modified:
            with open(CONFIG_PATH, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Configuration updated successfully")
        else:
            print(f"Current configuration:")
            for key, value in config.items():
                if key.endswith("_api_key") and value:
                    print(f"{key}: {'*' * 10}")
                else:
                    print(f"{key}: {value}")
    
    elif args.command == "index":
        # Index document
        simple_rag = SimpleRAG()
        success = simple_rag.index_document(args.file_path)
        
        if success:
            print(f"Document indexed successfully: {args.file_path}")
        else:
            print(f"Failed to index document: {args.file_path}")
            sys.exit(1)
    
    elif args.command == "query":
        # Query documents
        simple_rag = SimpleRAG()
        answer = simple_rag.query(args.question)
        print("\nAnswer:")
        print("-------")
        print(answer)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()