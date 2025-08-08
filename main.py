import asyncio
from mcp_server import main as mcp_main


def main():
    print("Starting AnyDB MCP Server...")
    print("This is an MCP (Model Context Protocol) server that provides database tools.")
    print("Available tools:")
    print("  - query_entity: Query tables with natural language")
    print("  - insert_entity: Insert data into tables")
    print("  - update_entity: Update table records")
    print("  - delete_entity: Delete table records")
    print("  - create_table: Create new tables")
    print("  - sql_query: Execute raw SELECT queries")
    print("  - sql_execute: Execute raw modification queries")
    print("\nConnecting to Ollama at localhost:1434 with llama3.1 model")
    print("Make sure Ollama is running with: ollama serve --port 1434")
    print("\nRunning MCP server via stdio...")
    
    asyncio.run(mcp_main())


if __name__ == "__main__":
    main()
