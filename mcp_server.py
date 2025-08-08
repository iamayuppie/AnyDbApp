import asyncio
import json
import sqlite3
import os
import logging
import hashlib
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import ollama
from mcp.server import Server
from mcp import Tool
import aiosqlite

# Optional vector database imports with fallback
VECTOR_DB_AVAILABLE = True
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    import tiktoken
except ImportError as e:
    VECTOR_DB_AVAILABLE = False
    logger_fallback_msg = f"Vector database dependencies not available: {e}"
    print(f"WARNING: {logger_fallback_msg}")
    print("Vector database tools will be disabled. Run 'pip install -r requirements.txt' to enable them.")

# Set up logging
def setup_logging():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "mcp_server.log")
    
    # Create logger
    logger = logging.getLogger('mcp_server')
    logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()
logger.info("="*50)
logger.info("MCP Server Starting Up")
logger.info("="*50)


class DatabaseManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use absolute path to ensure database is created in the same directory as the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(script_dir, "anydb.sqlite")
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.close()
        logger.info(f"Database initialized at: {self.db_path}")
        print(f"Database initialized at: {self.db_path}")
    
    async def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        logger.debug(f"Executing query: {query} with params: {params}")
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                result = [dict(row) for row in rows]
                logger.debug(f"Query returned {len(result)} rows")
                return result
    
    async def execute_modify(self, query: str, params: tuple = ()) -> int:
        logger.debug(f"Executing modify: {query} with params: {params}")
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                await db.commit()
                rowcount = cursor.rowcount
                logger.debug(f"Modify operation affected {rowcount} rows")
                return rowcount


class OllamaClient:
    def __init__(self, host: str = "localhost", port: int = 1434, model: str = "llama3.1"):
        self.client = ollama.Client(host=f"http://{host}:{port}")
        self.model = model
    
    async def generate_sql(self, instruction: str, table_name: str, schema: Optional[str] = None) -> str:
        prompt = f"""
        Generate a SQL query for the following instruction:
        Instruction: {instruction}
        Table: {table_name}
        """
        
        if schema:
            prompt += f"\nTable Schema: {schema}"
        
        prompt += "\nReturn only the SQL query, no explanations."
        
        logger.debug(f"Generating SQL with Ollama - Instruction: {instruction}, Table: {table_name}")
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            sql_query = response["message"]["content"].strip()
            logger.debug(f"Ollama generated SQL: {sql_query}")
            return sql_query
        except Exception as e:
            logger.error(f"Ollama error: {str(e)}")
            raise Exception(f"Ollama error: {str(e)}")


class VectorDatabaseManager:
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


# MCP Server Implementation
server = Server("anydb-mcp")
db_manager = DatabaseManager()
ollama_client = OllamaClient()

# Initialize vector database manager with error handling
try:
    vector_db_manager = VectorDatabaseManager()
except Exception as e:
    logger.error(f"Failed to initialize vector database: {str(e)}")
    # Create a dummy manager that always returns unavailable
    class DummyVectorManager:
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
    
    vector_db_manager = DummyVectorManager()

@server.list_tools()
async def list_tools() -> List[Tool]:
    # Core database tools (always available)
    tools = [
        Tool(
            name="query_entity",
            description="Query any entity/table with natural language instructions",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Name of the entity/table to query"},
                    "instruction": {"type": "string", "description": "Natural language query instruction", "default": "SELECT all records"}
                },
                "required": ["entity_name"]
            }
        ),
        Tool(
            name="insert_entity",
            description="Insert records into any entity/table",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Name of the entity/table"},
                    "data": {"type": "string", "description": "Data to insert (JSON or description)"}
                },
                "required": ["entity_name", "data"]
            }
        ),
        Tool(
            name="update_entity",
            description="Update records in any entity/table",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Name of the entity/table"},
                    "instruction": {"type": "string", "description": "Update instruction"},
                    "conditions": {"type": "string", "description": "WHERE conditions", "default": ""}
                },
                "required": ["entity_name", "instruction"]
            }
        ),
        Tool(
            name="delete_entity",
            description="Delete records from any entity/table",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Name of the entity/table"},
                    "conditions": {"type": "string", "description": "WHERE conditions for deletion", "default": ""}
                },
                "required": ["entity_name"]
            }
        ),
        Tool(
            name="create_table",
            description="Create a new table/entity with specified schema",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Name of the new table"},
                    "schema_description": {"type": "string", "description": "Description of table schema"}
                },
                "required": ["entity_name", "schema_description"]
            }
        ),
        Tool(
            name="sql_query",
            description="Execute raw SQL SELECT queries",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"}
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="sql_execute",
            description="Execute raw SQL modification queries",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"}
                },
                "required": ["query"]
            }
        )
    ]
    
    # Add vector database tools only if available
    if getattr(vector_db_manager, 'available', False):
        vector_tools = [
            Tool(
                name="add_file_to_vector_db",
                description="Add a file to the vector database for semantic search and RAG",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Name of the file"},
                        "content": {"type": "string", "description": "Content of the file (text)"},
                        "metadata": {"type": "object", "description": "Optional metadata for the file", "default": {}}
                    },
                    "required": ["filename", "content"]
                }
            ),
            Tool(
                name="search_vector_db",
                description="Search the vector database for relevant file content based on semantic similarity",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query for semantic similarity"},
                        "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 5}
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="list_vector_files",
                description="List all files stored in the vector database",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            Tool(
                name="remove_file_from_vector_db",
                description="Remove a file from the vector database",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Name of the file to remove"}
                    },
                    "required": ["filename"]
                }
            )
        ]
        tools.extend(vector_tools)
    
    return tools

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    logger.info(f"INCOMING REQUEST - Tool: {name}")
    logger.info(f"ARGUMENTS: {json.dumps(arguments, indent=2)}")
    
    try:
        if name == "query_entity":
            entity_name = arguments["entity_name"]
            instruction = arguments.get("instruction", "SELECT all records")
            logger.info(f"Query Entity - Table: {entity_name}, Instruction: {instruction}")
            
            sql_query = await ollama_client.generate_sql(instruction, entity_name)
            results = await db_manager.execute_query(sql_query)
            
            response_data = {"sql": sql_query, "results": results}
            response = [{"type": "text", "text": json.dumps(response_data, indent=2)}]
            logger.info(f"RESPONSE: Query returned {len(results)} rows")
            logger.debug(f"FULL RESPONSE: {json.dumps(response_data, indent=2)}")
            return response
        
        elif name == "insert_entity":
            entity_name = arguments["entity_name"]
            data = arguments["data"]
            instruction = f"INSERT into {entity_name} with data: {data}"
            sql_query = await ollama_client.generate_sql(instruction, entity_name)
            affected_rows = await db_manager.execute_modify(sql_query)
            return [{"type": "text", "text": json.dumps({"sql": sql_query, "affected_rows": affected_rows})}]
        
        elif name == "update_entity":
            entity_name = arguments["entity_name"]
            instruction = arguments["instruction"]
            conditions = arguments.get("conditions", "")
            full_instruction = f"UPDATE {entity_name} {instruction}"
            if conditions:
                full_instruction += f" WHERE {conditions}"
            sql_query = await ollama_client.generate_sql(full_instruction, entity_name)
            affected_rows = await db_manager.execute_modify(sql_query)
            return [{"type": "text", "text": json.dumps({"sql": sql_query, "affected_rows": affected_rows})}]
        
        elif name == "delete_entity":
            entity_name = arguments["entity_name"]
            conditions = arguments.get("conditions", "")
            instruction = f"DELETE FROM {entity_name}"
            if conditions:
                instruction += f" WHERE {conditions}"
            sql_query = await ollama_client.generate_sql(instruction, entity_name)
            affected_rows = await db_manager.execute_modify(sql_query)
            return [{"type": "text", "text": json.dumps({"sql": sql_query, "affected_rows": affected_rows})}]
        
        elif name == "create_table":
            entity_name = arguments["entity_name"]
            schema_description = arguments["schema_description"]
            instruction = f"CREATE TABLE {entity_name} with schema: {schema_description}"
            logger.info(f"Create Table - Name: {entity_name}, Schema: {schema_description}")
            
            sql_query = await ollama_client.generate_sql(instruction, entity_name)
            await db_manager.execute_modify(sql_query)
            
            response_data = {
                "sql": sql_query, 
                "message": f"Table {entity_name} created successfully", 
                "database_path": db_manager.db_path
            }
            response = [{"type": "text", "text": json.dumps(response_data)}]
            logger.info(f"RESPONSE: Table {entity_name} created successfully")
            logger.debug(f"FULL RESPONSE: {json.dumps(response_data, indent=2)}")
            return response
        
        elif name == "sql_query":
            query = arguments["query"]
            results = await db_manager.execute_query(query)
            return [{"type": "text", "text": json.dumps(results, indent=2)}]
        
        elif name == "sql_execute":
            query = arguments["query"]
            affected_rows = await db_manager.execute_modify(query)
            return [{"type": "text", "text": f"Operation completed. Affected rows: {affected_rows}"}]
        
        elif name == "add_file_to_vector_db":
            if not getattr(vector_db_manager, 'available', False):
                error_msg = "Vector database not available. Please install dependencies: pip install sentence-transformers"
                return [{"type": "text", "text": json.dumps({"error": error_msg, "status": "disabled"})}]
                
            filename = arguments["filename"]
            content = arguments["content"]
            metadata = arguments.get("metadata", {})
            metadata["added_date"] = datetime.now().isoformat()
            
            logger.info(f"Adding file to vector database - Filename: {filename}, Content length: {len(content)}")
            
            result = await vector_db_manager.add_file(filename, content, metadata)
            response = [{"type": "text", "text": json.dumps(result, indent=2)}]
            
            if result.get("status") == "success":
                logger.info(f"RESPONSE: File added successfully - {result['chunks_added']} chunks")
            else:
                logger.warning(f"RESPONSE: File add failed - {result}")
            return response
        
        elif name == "search_vector_db":
            if not getattr(vector_db_manager, 'available', False):
                error_msg = "Vector database not available. Please install dependencies: pip install sentence-transformers"
                return [{"type": "text", "text": json.dumps({"error": error_msg, "status": "disabled"})}]
                
            query = arguments["query"]
            max_results = arguments.get("max_results", 5)
            
            logger.info(f"Searching vector database - Query: {query}, Max results: {max_results}")
            
            search_results = await vector_db_manager.search_files(query, max_results)
            response_data = {
                "query": query,
                "results": search_results,
                "total_results": len(search_results)
            }
            response = [{"type": "text", "text": json.dumps(response_data, indent=2)}]
            logger.info(f"RESPONSE: Found {len(search_results)} relevant results")
            return response
        
        elif name == "list_vector_files":
            if not getattr(vector_db_manager, 'available', False):
                error_msg = "Vector database not available. Please install dependencies: pip install sentence-transformers"
                return [{"type": "text", "text": json.dumps({"error": error_msg, "files": [], "total_files": 0})}]
                
            logger.info("Listing files in vector database")
            
            files = await vector_db_manager.list_files()
            response_data = {
                "files": files,
                "total_files": len(files)
            }
            response = [{"type": "text", "text": json.dumps(response_data, indent=2)}]
            logger.info(f"RESPONSE: Listed {len(files)} files")
            return response
        
        elif name == "remove_file_from_vector_db":
            if not getattr(vector_db_manager, 'available', False):
                error_msg = "Vector database not available. Please install dependencies: pip install sentence-transformers"
                return [{"type": "text", "text": json.dumps({"error": error_msg, "status": "disabled"})}]
                
            filename = arguments["filename"]
            
            logger.info(f"Removing file from vector database - Filename: {filename}")
            
            chunks_removed = await vector_db_manager.remove_file(filename)
            response_data = {
                "filename": filename,
                "chunks_removed": chunks_removed,
                "status": "success" if chunks_removed > 0 else "no_file_found"
            }
            response = [{"type": "text", "text": json.dumps(response_data, indent=2)}]
            logger.info(f"RESPONSE: Removed {chunks_removed} chunks")
            return response
        
        else:
            logger.error(f"Unknown tool requested: {name}")
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(f"TOOL ERROR - Tool: {name}, Error: {str(e)}")
        logger.error(f"ERROR RESPONSE: {error_msg}")
        return [{"type": "text", "text": error_msg}]

async def main():
    logger.info("Starting MCP server stdio connection...")
    from mcp.server.stdio import stdio_server
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP server connected and ready")
            await server.run(read_stream, write_stream, server.create_initialization_options())
    except BaseExceptionGroup as eg:
        # Handle ExceptionGroup (TaskGroup errors) - Python 3.11+
        logger.error(f"MCP server ExceptionGroup with {len(eg.exceptions)} exceptions:")
        for i, exc in enumerate(eg.exceptions):
            logger.error(f"Exception {i+1}: {type(exc).__name__}: {str(exc)}")
            import traceback
            logger.error(f"Traceback {i+1}: {''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}")
        
        # Check if these are non-critical background task errors
        non_critical_errors = ['RuntimeError', 'OSError', 'ConnectionError']
        all_non_critical = all(type(exc).__name__ in non_critical_errors for exc in eg.exceptions)
        
        if all_non_critical:
            logger.warning("All exceptions appear to be non-critical background task errors - continuing operation")
        else:
            logger.error("Critical exceptions detected in ExceptionGroup - this may affect functionality")
            # Don't raise for now to avoid crashing the server, but log it prominently
            logger.error("SERVER CONTINUING DESPITE CRITICAL ERRORS - monitor for functionality issues")
    except Exception as e:
        logger.error(f"MCP server error: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    logger.info("Running MCP server as main process")
    asyncio.run(main())