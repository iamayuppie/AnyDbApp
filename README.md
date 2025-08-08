# AnyDB MCP Server

A Model Context Protocol (MCP) server that provides intelligent database operations through natural language processing. This server integrates SQLite databases with Ollama for AI-powered SQL generation and execution.

## Features

### Core Database Operations
- **Natural Language to SQL**: Convert plain English instructions into SQL queries using Ollama
- **Universal Database Operations**: Works with any SQLite table/entity without predefined schemas
- **MCP Integration**: Seamlessly integrates with Claude Desktop and other MCP-compatible clients
- **Async Operations**: Built on modern Python async/await for high performance
- **Safety First**: Separate tools for read and write operations

### Vector Database & RAG (NEW!)
- **File Embedding**: Automatically convert files into vector embeddings for semantic search
- **Semantic Search**: Find relevant content using natural language queries instead of exact keyword matching
- **RAG Support**: Enable Claude Desktop to answer questions about uploaded documents with context
- **Smart Chunking**: Intelligently splits large documents into overlapping chunks for better retrieval
- **Persistent Storage**: ChromaDB-powered vector database with automatic embedding generation

## Available Tools

### Database Tools

#### 1. `query_entity`
Query any table with natural language instructions.

**Parameters**: 
- `entity_name` (required): Name of the table to query
- `instruction` (optional): Natural language query instruction (default: "SELECT all records")

**Example**: Query users table for active accounts

#### 2. `insert_entity`
Insert records into any table using natural language descriptions.

**Parameters**:
- `entity_name` (required): Name of the table
- `data` (required): Data to insert (JSON or natural description)

**Example**: Insert a new user with email and name

#### 3. `update_entity`
Update records in any table with conditions.

**Parameters**:
- `entity_name` (required): Name of the table
- `instruction` (required): Update instruction
- `conditions` (optional): WHERE conditions

**Example**: Update user status to active where email matches

#### 4. `delete_entity`
Delete records from any table with optional conditions.

**Parameters**:
- `entity_name` (required): Name of the table
- `conditions` (optional): WHERE conditions for deletion

**Example**: Delete inactive users older than 30 days

#### 5. `create_table`
Create new tables with AI-generated schemas.

**Parameters**:
- `entity_name` (required): Name of the new table
- `schema_description` (required): Description of table schema

**Example**: Create a products table with name, price, and category

#### 6. `sql_query`
Execute raw SQL SELECT queries directly.

**Parameters**:
- `query` (required): SQL query to execute

**Example**: Direct SQL for complex joins and analytics

#### 7. `sql_execute`
Execute raw SQL modification queries (INSERT, UPDATE, DELETE, CREATE, etc.).

**Parameters**:
- `query` (required): SQL query to execute

**Example**: Direct SQL for complex data modifications

### Vector Database Tools (NEW!)

#### 8. `add_file_to_vector_db`
Add a file to the vector database for semantic search and RAG (Retrieval Augmented Generation).

**Parameters**:
- `filename` (required): Name of the file
- `content` (required): Content of the file (text)
- `metadata` (optional): Optional metadata for the file

**Example**: Add a document about machine learning for later semantic search

#### 9. `search_vector_db`
Search the vector database for relevant file content using semantic similarity.

**Parameters**:
- `query` (required): Search query for semantic similarity
- `max_results` (optional): Maximum number of results to return (default: 5)

**Example**: Find documents related to "neural networks and AI" 

#### 10. `list_vector_files`
List all files stored in the vector database.

**Parameters**: None

**Example**: View all documents available for search

#### 11. `remove_file_from_vector_db`
Remove a file from the vector database.

**Parameters**:
- `filename` (required): Name of the file to remove

**Example**: Delete outdated documents from the knowledge base

## Installation

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) running locally
- Claude Desktop (for MCP integration)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/iamayuppie/AnyDbApp.git
cd AnyDbApp
```

2. **Install dependencies (choose one option):**

**Option A: Full installation (recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Minimal installation (core database tools only)**  
```bash
pip install -r requirements-minimal.txt
```

**Option C: Manual installation of core dependencies**
```bash
pip install mcp aiosqlite ollama
```

2. **Start Ollama:**
```bash
ollama serve --port 1434
ollama pull llama3.1  # or your preferred model
```

3. **Run the server:**
```bash
python main.py
```

## Claude Desktop Integration

Add this server to Claude Desktop by editing your config file:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "anydb": {
      "command": "python",
      "args": ["C:\\path\\to\\AnyDbApp\\main.py"],
      "env": {
        "PYTHONPATH": "C:\\path\\to\\AnyDbApp"
      }
    }
  }
}
```

Restart Claude Desktop to connect the server.

## Configuration

### Ollama Settings
Default configuration in `mcp_server.py`:
- **Host**: localhost
- **Port**: 1434  
- **Model**: llama3.1

### Database Settings
- **Default DB**: `anydb.sqlite` (created automatically)
- **Location**: Same directory as the server
- **Type**: SQLite with foreign key constraints enabled

## Usage Examples

Once integrated with Claude Desktop, you can use natural language:

### Database Operations
- *"Create a users table with id, name, email, and created_at fields"*
- *"Show me all active users from the last 30 days"*
- *"Insert a new product: iPhone 15, price $999, category Electronics"*
- *"Update all pending orders to processed where amount > 100"*
- *"Delete test users where email contains 'test'"*

### Vector Database & File Operations
- *"Add this document to the knowledge base"* (when attaching a file in Claude Desktop)
- *"Search for information about machine learning algorithms"*
- *"Find documents related to user authentication and security"*
- *"What does the uploaded contract say about payment terms?"*
- *"Show me all documents I've added to the database"*
- *"Remove the old privacy policy document"*

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Claude        │────│  MCP Server  │────│    Ollama       │
│   Desktop       │    │   (stdio)    │    │   (localhost)   │
│  + File Upload  │    │              │    │                 │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Dual Storage   │
                    │                  │
                    │ ┌──────────────┐ │
                    │ │   SQLite     │ │  ← Structured Data
                    │ │   Database   │ │
                    │ └──────────────┘ │
                    │                  │
                    │ ┌──────────────┐ │
                    │ │  ChromaDB    │ │  ← Document Embeddings
                    │ │ Vector Store │ │     & Semantic Search
                    │ └──────────────┘ │
                    └──────────────────┘
```

## Development

### Project Structure
```
AnyDbApp/
├── main.py              # Clean entry point with startup info
├── mcp_server.py        # MCP server setup and tool routing
├── dbtool.py            # Database operations and SQL tools
├── filetool.py          # Vector database and file operations
├── requirements.txt     # Python dependencies  
├── requirements-minimal.txt  # Core dependencies only
├── pyproject.toml      # Project metadata
└── README.md           # This file
```

### Key Components

**Core Modules:**
- **main.py**: Entry point with dependency checking and startup information
- **mcp_server.py**: MCP protocol implementation, tool registration, and request routing
- **dbtool.py**: Database operations, SQL generation, and data management
- **filetool.py**: Vector database operations, file processing, and semantic search

**Business Logic Classes:**
- **DatabaseManager**: Handles async SQLite operations and database connections
- **DatabaseTools**: High-level database operations with natural language support
- **OllamaClient**: Manages AI model communication for SQL generation
- **VectorDatabaseManager**: Manages ChromaDB operations and document embeddings  
- **FileTools**: High-level file operations and semantic search functionality

## Troubleshooting

### Common Issues

1. **Server won't start**: Check if Ollama is running on port 1434
2. **No tools showing in Claude**: Verify MCP config path and restart Claude Desktop  
3. **SQL errors**: Check table names and ensure proper natural language descriptions
4. **Ollama connection failed**: Confirm Ollama model is installed and accessible

### Debug Mode
Run with Python's verbose mode for detailed logs:
```bash
python -v main.py
```

## License

This project is open source. See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review Ollama and MCP documentation
- Open an issue on the repository