import asyncio
import json
import sqlite3
import os
from typing import Any, Dict, List, Optional, Union

import ollama
from mcp.server import Server
from mcp import Tool
import aiosqlite


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
        print(f"Database initialized at: {self.db_path}")
    
    async def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def execute_modify(self, query: str, params: tuple = ()) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                await db.commit()
                return cursor.rowcount


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
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"].strip()
        except Exception as e:
            raise Exception(f"Ollama error: {str(e)}")


# MCP Server Implementation
server = Server("anydb-mcp")
db_manager = DatabaseManager()
ollama_client = OllamaClient()

@server.list_tools()
async def list_tools() -> List[Tool]:
    return [
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

@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        if name == "query_entity":
            entity_name = arguments["entity_name"]
            instruction = arguments.get("instruction", "SELECT all records")
            sql_query = await ollama_client.generate_sql(instruction, entity_name)
            results = await db_manager.execute_query(sql_query)
            return [{"type": "text", "text": json.dumps({"sql": sql_query, "results": results}, indent=2)}]
        
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
            print(f"Creating table: {entity_name} with instruction: {instruction}")
            sql_query = await ollama_client.generate_sql(instruction, entity_name)
            print(f"Generated SQL: {sql_query}")
            await db_manager.execute_modify(sql_query)
            print(f"Table {entity_name} created successfully in database: {db_manager.db_path}")
            return [{"type": "text", "text": json.dumps({"sql": sql_query, "message": f"Table {entity_name} created successfully", "database_path": db_manager.db_path})}]
        
        elif name == "sql_query":
            query = arguments["query"]
            results = await db_manager.execute_query(query)
            return [{"type": "text", "text": json.dumps(results, indent=2)}]
        
        elif name == "sql_execute":
            query = arguments["query"]
            affected_rows = await db_manager.execute_modify(query)
            return [{"type": "text", "text": f"Operation completed. Affected rows: {affected_rows}"}]
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        return [{"type": "text", "text": f"Error: {str(e)}"}]

async def main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())