"""
AnyDB MCP Server - Main Entry Point

A Model Context Protocol server that provides intelligent database operations
and file/document management with semantic search capabilities.

Usage:
    python main.py

Requirements:
    - Ollama running on localhost:1434
    - Claude Desktop configured with this server
"""

import asyncio
import sys
import os

# Add current directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server import main as mcp_main


def print_startup_info():
    """Display startup information and system status."""
    print("=" * 60)
    print("🚀 AnyDB MCP Server")
    print("=" * 60)
    print("Model Context Protocol server for database and file operations")
    print()
    
    # Check dependencies
    print("📋 Available Tools:")
    print("   Database Tools:")
    print("   • query_entity     - Query tables with natural language")
    print("   • insert_entity    - Insert data into tables")
    print("   • update_entity    - Update table records")
    print("   • delete_entity    - Delete table records")
    print("   • create_table     - Create new tables")
    print("   • sql_query        - Execute raw SELECT queries")
    print("   • sql_execute      - Execute raw modification queries")
    print()
    
    print("   Vector Database Tools:")
    print("   • add_file_to_vector_db     - Add files for semantic search")
    print("   • search_vector_db          - Search files by meaning")
    print("   • list_vector_files         - List stored files")
    print("   • remove_file_from_vector_db - Remove files")
    
    print()
    print("🔗 Prerequisites:")
    print("   • Ollama server running on localhost:1434")
    print("   • Start with: ollama serve --port 1434")
    print("   • Model: llama3.1 (or configure in dbtool.py)")
    print()
    print("🖥️  Integration:")
    print("   • Add this server to Claude Desktop MCP configuration")
    print("   • Server will communicate via stdio")
    print("   • Logs written to: mcp_server.log")
    print()
    print("Starting server...")
    print("-" * 60)


def main():
    """Main entry point for the AnyDB MCP Server."""
    try:
        # Only show startup info when running directly (not via Claude Desktop)
        # Claude Desktop communicates via stdio and expects clean JSON
        if os.getenv('MCP_DISABLE_STARTUP_MESSAGES') != '1' and sys.stdin.isatty():
            print_startup_info()
        
        asyncio.run(mcp_main())
    except KeyboardInterrupt:
        # Only print if we have a terminal (not when running via Claude Desktop)
        if sys.stdin.isatty():
            print("\n⛔ Server stopped by user")
        sys.exit(0)
    except Exception as e:
        # Log errors but don't print to stdout when running via Claude Desktop
        if sys.stdin.isatty():
            print(f"\n❌ Server failed to start: {str(e)}")
            print("Check mcp_server.log for detailed error information")
        # Always log to file for debugging
        import logging
        logger = logging.getLogger('mcp_server')
        logger.error(f"Server failed to start: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
