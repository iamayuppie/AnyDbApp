"""
MCP Server for AnyDB - Model Context Protocol server implementation.

This module contains only the MCP server setup, tool definitions, and request handling.
Business logic is separated into dbtool.py and filetool.py modules.
"""

import asyncio
import json
import os
import logging
from datetime import datetime
from typing import Any, Dict, List

from mcp.server import Server
from mcp import Tool

# Import our business logic modules
from dbtool import DatabaseManager, OllamaClient, DatabaseTools
from filetool import VectorDatabaseManager, FileTools, DummyVectorManager

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

# Initialize MCP Server
server = Server("anydb-mcp")

# Initialize business logic components
db_manager = DatabaseManager()
ollama_client = OllamaClient()
db_tools = DatabaseTools(db_manager, ollama_client)

# Initialize vector database manager with error handling
try:
    vector_db_manager = VectorDatabaseManager()
    file_tools = FileTools(vector_db_manager)
except Exception as e:
    logger.error(f"Failed to initialize vector database: {str(e)}")
    vector_db_manager = DummyVectorManager()
    file_tools = FileTools(vector_db_manager)


@server.list_tools()
async def list_tools() -> List[Tool]:
    """Define all available MCP tools."""
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
                    "data": {"type": "string", "description": "Data to insert (JSON or natural description)"}
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
    """Handle MCP tool calls and delegate to appropriate business logic modules."""
    logger.info(f"INCOMING REQUEST - Tool: {name}")
    logger.info(f"ARGUMENTS: {json.dumps(arguments, indent=2)}")
    
    try:
        # Database tools
        if name == "query_entity":
            entity_name = arguments["entity_name"]
            instruction = arguments.get("instruction", "SELECT all records")
            result = await db_tools.query_entity(entity_name, instruction)
            response = [{"type": "text", "text": json.dumps(result, indent=2)}]
            logger.debug(f"FULL RESPONSE: {json.dumps(result, indent=2)}")
            return response
        
        elif name == "insert_entity":
            entity_name = arguments["entity_name"]
            data = arguments["data"]
            result = await db_tools.insert_entity(entity_name, data)
            response = [{"type": "text", "text": json.dumps(result, indent=2)}]
            logger.debug(f"FULL RESPONSE: {json.dumps(result, indent=2)}")
            return response
        
        elif name == "update_entity":
            entity_name = arguments["entity_name"]
            instruction = arguments["instruction"]
            conditions = arguments.get("conditions", "")
            result = await db_tools.update_entity(entity_name, instruction, conditions)
            response = [{"type": "text", "text": json.dumps(result, indent=2)}]
            logger.debug(f"FULL RESPONSE: {json.dumps(result, indent=2)}")
            return response
        
        elif name == "delete_entity":
            entity_name = arguments["entity_name"]
            conditions = arguments.get("conditions", "")
            result = await db_tools.delete_entity(entity_name, conditions)
            response = [{"type": "text", "text": json.dumps(result, indent=2)}]
            logger.debug(f"FULL RESPONSE: {json.dumps(result, indent=2)}")
            return response
        
        elif name == "create_table":
            entity_name = arguments["entity_name"]
            schema_description = arguments["schema_description"]
            result = await db_tools.create_table(entity_name, schema_description)
            response = [{"type": "text", "text": json.dumps(result)}]
            logger.debug(f"FULL RESPONSE: {json.dumps(result, indent=2)}")
            return response
        
        elif name == "sql_query":
            query = arguments["query"]
            result = await db_tools.execute_raw_query(query)
            response = [{"type": "text", "text": json.dumps(result, indent=2)}]
            logger.debug(f"FULL RESPONSE: {json.dumps(result, indent=2)}")
            return response
        
        elif name == "sql_execute":
            query = arguments["query"]
            result = await db_tools.execute_raw_modify(query)
            response = [{"type": "text", "text": json.dumps(result, indent=2)}]
            logger.debug(f"FULL RESPONSE: {json.dumps(result, indent=2)}")
            return response
        
        # Vector database tools
        elif name == "add_file_to_vector_db":
            filename = arguments["filename"]
            content = arguments["content"]
            metadata = arguments.get("metadata", {})
            result = await file_tools.add_file_to_vector_db(filename, content, metadata)
            response = [{"type": "text", "text": json.dumps(result, indent=2)}]
            logger.debug(f"FULL RESPONSE: {json.dumps(result, indent=2)}")
            return response
        
        elif name == "search_vector_db":
            query = arguments["query"]
            max_results = arguments.get("max_results", 5)
            result = await file_tools.search_vector_db(query, max_results)
            response = [{"type": "text", "text": json.dumps(result, indent=2)}]
            logger.debug(f"FULL RESPONSE: {json.dumps(result, indent=2)}")
            return response
        
        elif name == "list_vector_files":
            result = await file_tools.list_vector_files()
            response = [{"type": "text", "text": json.dumps(result, indent=2)}]
            logger.debug(f"FULL RESPONSE: {json.dumps(result, indent=2)}")
            return response
        
        elif name == "remove_file_from_vector_db":
            filename = arguments["filename"]
            result = await file_tools.remove_file_from_vector_db(filename)
            response = [{"type": "text", "text": json.dumps(result, indent=2)}]
            logger.debug(f"FULL RESPONSE: {json.dumps(result, indent=2)}")
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
    """Main MCP server entry point."""
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
            logger.error("SERVER CONTINUING DESPITE CRITICAL ERRORS - monitor for functionality issues")
    except Exception as e:
        logger.error(f"MCP server error: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    logger.info("Running MCP server as main process")
    asyncio.run(main())