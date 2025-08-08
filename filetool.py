"""
File and vector database operations module for AnyDB MCP Server.

Contains all file and vector database functionality including:
- ChromaDB vector database management
- Document embedding and chunking
- Semantic search operations
- File management for RAG (Retrieval Augmented Generation)
"""

import os
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# Optional vector database imports with fallback
VECTOR_DB_AVAILABLE = True
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    import tiktoken
except ImportError as e:
    VECTOR_DB_AVAILABLE = False
    print(f"WARNING: Vector database dependencies not available: {e}")
    print("Vector database tools will be disabled. Run 'pip install sentence-transformers' to enable them.")

# Get logger
logger = logging.getLogger('mcp_server')


class VectorDatabaseManager:
    """Manages ChromaDB vector database operations for semantic search."""
    
    def __init__(self, collection_name: str = "file_embeddings"):
        if not VECTOR_DB_AVAILABLE:
            logger.warning("Vector database dependencies not available - vector operations will be disabled")
            self.available = False
            return
            
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(script_dir, "vector_db")
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize sentence transformer for embeddings
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize tokenizer for text chunking
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
            self.available = True
            logger.info(f"Vector database initialized at: {self.db_path}")
            logger.info(f"Collection '{collection_name}' ready with {self.collection.count()} documents")
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            self.available = False
    
    def _chunk_text(self, text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks based on token count."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if i + max_tokens >= len(tokens):
                break
        
        return chunks
    
    def _generate_file_id(self, filename: str, content: str) -> str:
        """Generate a unique ID for a file based on filename and content hash."""
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
        return f"{filename}_{content_hash}"
    
    async def add_file(self, filename: str, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a file to the vector database with embeddings."""
        if not getattr(self, 'available', False):
            return {"error": "Vector database not available", "status": "disabled"}
            
        logger.info(f"Adding file to vector database: {filename}")
        
        if metadata is None:
            metadata = {}
        
        # Generate unique file ID
        file_id = self._generate_file_id(filename, content)
        
        # Remove existing entries for this file
        await self.remove_file(filename)
        
        # Chunk the content
        chunks = self._chunk_text(content)
        logger.debug(f"Split file into {len(chunks)} chunks")
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_id}_chunk_{i}"
            chunk_metadata = {
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "file_id": file_id,
                **metadata
            }
            
            documents.append(chunk)
            metadatas.append(chunk_metadata)
            ids.append(chunk_id)
        
        # Add to ChromaDB (embeddings are generated automatically)
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        except Exception as e:
            logger.error(f"ChromaDB add operation failed: {str(e)}")
            import traceback
            logger.error(f"ChromaDB traceback: {traceback.format_exc()}")
            raise
        
        logger.info(f"File '{filename}' added with {len(chunks)} chunks")
        return {
            "filename": filename,
            "file_id": file_id,
            "chunks_added": len(chunks),
            "status": "success"
        }
    
    async def remove_file(self, filename: str) -> int:
        """Remove all chunks of a file from the vector database."""
        if not getattr(self, 'available', False):
            return 0
            
        try:
            # Find all chunks for this file
            results = self.collection.get(
                where={"filename": filename}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Removed {len(results['ids'])} chunks for file '{filename}'")
                return len(results['ids'])
            else:
                logger.debug(f"No chunks found for file '{filename}'")
                return 0
        except Exception as e:
            logger.error(f"Error removing file '{filename}': {str(e)}")
            return 0
    
    async def search_files(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant file chunks based on semantic similarity."""
        if not getattr(self, 'available', False):
            return []
            
        logger.debug(f"Searching vector database for: {query}")
        
        # Search the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=min(max_results, self.collection.count())
        )
        
        # Format results
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                result = {
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity_score": 1 - results['distances'][0][i],  # Convert distance to similarity
                    "id": results['ids'][0][i]
                }
                search_results.append(result)
        
        logger.debug(f"Found {len(search_results)} relevant chunks")
        return search_results
    
    async def list_files(self) -> List[Dict[str, Any]]:
        """List all files in the vector database."""
        if not getattr(self, 'available', False):
            return []
            
        try:
            # Get all unique filenames
            all_results = self.collection.get()
            
            files = {}
            for metadata in all_results['metadatas']:
                filename = metadata['filename']
                if filename not in files:
                    files[filename] = {
                        "filename": filename,
                        "file_id": metadata['file_id'],
                        "total_chunks": metadata['total_chunks'],
                        "added_date": metadata.get('added_date', 'unknown')
                    }
            
            return list(files.values())
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return []


class FileTools:
    """High-level file and vector database tool operations."""
    
    def __init__(self, vector_db_manager: VectorDatabaseManager):
        self.vector_db_manager = vector_db_manager
    
    async def add_file_to_vector_db(self, filename: str, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add a file to the vector database for semantic search and RAG."""
        if not getattr(self.vector_db_manager, 'available', False):
            return {"error": "Vector database not available", "status": "disabled"}
        
        if metadata is None:
            metadata = {}
        metadata["added_date"] = datetime.now().isoformat()
        
        logger.info(f"Adding file to vector database - Filename: {filename}, Content length: {len(content)}")
        
        result = await self.vector_db_manager.add_file(filename, content, metadata)
        
        if result.get("status") == "success":
            logger.info(f"File added successfully - {result['chunks_added']} chunks")
        else:
            logger.warning(f"File add failed - {result}")
        
        return result
    
    async def search_vector_db(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search the vector database for relevant file content using semantic similarity."""
        if not getattr(self.vector_db_manager, 'available', False):
            return {"error": "Vector database not available", "status": "disabled", "results": [], "total_results": 0}
        
        logger.info(f"Searching vector database - Query: {query}, Max results: {max_results}")
        
        search_results = await self.vector_db_manager.search_files(query, max_results)
        response_data = {
            "query": query,
            "results": search_results,
            "total_results": len(search_results)
        }
        
        logger.info(f"Found {len(search_results)} relevant results")
        return response_data
    
    async def list_vector_files(self) -> Dict[str, Any]:
        """List all files stored in the vector database."""
        if not getattr(self.vector_db_manager, 'available', False):
            return {"error": "Vector database not available", "files": [], "total_files": 0}
        
        logger.info("Listing files in vector database")
        
        files = await self.vector_db_manager.list_files()
        response_data = {
            "files": files,
            "total_files": len(files)
        }
        
        logger.info(f"Listed {len(files)} files")
        return response_data
    
    async def remove_file_from_vector_db(self, filename: str) -> Dict[str, Any]:
        """Remove a file from the vector database."""
        if not getattr(self.vector_db_manager, 'available', False):
            return {"error": "Vector database not available", "status": "disabled"}
        
        logger.info(f"Removing file from vector database - Filename: {filename}")
        
        chunks_removed = await self.vector_db_manager.remove_file(filename)
        response_data = {
            "filename": filename,
            "chunks_removed": chunks_removed,
            "status": "success" if chunks_removed > 0 else "no_file_found"
        }
        
        logger.info(f"Removed {chunks_removed} chunks")
        return response_data


# Dummy manager for when vector DB is not available
class DummyVectorManager:
    """Dummy vector manager when dependencies are not available."""
    
    def __init__(self):
        self.available = False
    
    async def add_file(self, *args, **kwargs):
        return {"error": "Vector database not available", "status": "disabled"}
    
    async def search_files(self, *args, **kwargs):
        return []
    
    async def list_files(self):
        return []
    
    async def remove_file(self, *args, **kwargs):
        return 0