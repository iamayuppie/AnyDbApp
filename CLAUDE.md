# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Starting the Server
```bash
# For development and testing (shows startup info)
python main.py

# For Claude Desktop integration (clean stdio output)
python mcp_server_stdio.py
```

### Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt
```

### Testing the Server
```bash
# Test Ollama connection (required for SQL generation)
ollama serve --port 1434
ollama pull llama3.1

# Check database and logs
ls -la anydb.sqlite vector_db/
tail -f mcp_server.log
```

## Architecture Overview

### Core Components

**Entry Points:**
- `main.py` - Development entry point with startup information and dependency checks
- `mcp_server_stdio.py` - Production entry point for Claude Desktop (clean stdio communication)

**Business Logic Separation:**
- `mcp_server.py` - MCP protocol implementation, tool registration, and request routing only
- `dbtool.py` - All database operations, SQL generation, and data management
- `filetool.py` - Vector database operations, file processing, and semantic search

### Data Flow Architecture

```
Claude Desktop → mcp_server_stdio.py → mcp_server.py → {dbtool.py | filetool.py}
                                          ↓              ↓
                                    DatabaseTools    FileTools
                                          ↓              ↓
                                    SQLite DB      ChromaDB
                                          ↓              ↓
                                     Ollama AI    Embeddings
```

### Key Design Patterns

**Graceful Degradation:** The server starts with core database tools and vector database

**Async/Await Throughout:** All operations use async/await for non-blocking I/O, essential for MCP server performance.

**Absolute Path Resolution:** Database and vector database paths use `os.path.abspath(__file__)` to ensure consistent file locations regardless of execution context.

**Comprehensive Logging:** All operations log to `mcp_server.log` with detailed request/response tracking for debugging Claude Desktop integration.

### Business Logic Classes

**Database Layer:**
- `DatabaseManager` - Async SQLite operations with proper connection handling
- `OllamaClient` - AI model communication for natural language to SQL conversion
- `DatabaseTools` - High-level database operations combining the above

**Vector Database Layer:**
- `VectorDatabaseManager` - ChromaDB operations with sentence transformers
- `FileTools` - High-level file operations and semantic search

### Tool Categories

**Database Tools (7 tools):**
- Natural language to SQL operations via Ollama
- Raw SQL execution for complex queries
- All operations return both SQL and results for transparency

**Vector Database Tools (4 tools):**
- File embedding with intelligent chunking using tiktoken
- Semantic search using sentence transformers
- RAG (Retrieval Augmented Generation) support for document Q&A

## Critical Implementation Details

### MCP Integration Requirements

**Claude Desktop Compatibility:** Use `mcp_server_stdio.py` for Claude Desktop integration. The main entry point produces console output that interferes with JSON communication.

**Tool Response Format:** All tools must return `List[Dict[str, Any]]` with `{"type": "text", "text": json_string}` format for MCP compatibility.

**Error Handling:** All errors are caught, logged, and returned as text responses to prevent MCP connection drops.

### Ollama Integration

**Default Configuration:**
- Host: localhost:1434
- Model: llama3.1
- All SQL generation goes through OllamaClient

**SQL Generation Pattern:** Natural language instructions are converted to SQL with table context, then executed on SQLite database.

### Vector Database Implementation

**Chunking Strategy:** Documents are split into 500-token chunks with 50-token overlap using tiktoken encoding for optimal retrieval.

**Embedding Model:** Uses `all-MiniLM-L6-v2` sentence transformer for consistent, fast embeddings.

**Search Results:** Returns similarity scores (1 - distance) and metadata for context.

### Database Paths

All databases use absolute paths relative to script location:
- SQLite: `{script_dir}/anydb.sqlite`
- ChromaDB: `{script_dir}/vector_db/`

### Logging Strategy

**File Logging Only:** All output goes to `mcp_server.log` to avoid interfering with MCP stdio communication.

**Log Levels Used:**
- INFO: Request/response tracking, major operations
- DEBUG: Query details, chunk counts, internal operations  
- ERROR: Failures, exceptions with full tracebacks

### Common Troubleshooting

**Ollama Connection Issues:** Check if Ollama is running on port 1434 and model is installed.

**Claude Desktop JSON Errors:** Ensure using `mcp_server_stdio.py` entry point, not `main.py`.

**Vector Database Unavailable:** logs the exception and exits out of the application.

**TaskGroup Errors:** These are non-critical background task errors that don't affect functionality.