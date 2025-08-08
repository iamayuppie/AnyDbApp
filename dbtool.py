"""
Database operations module for AnyDB MCP Server.

Contains all database-related functionality including:
- SQLite database management
- SQL query execution
- Natural language to SQL conversion via Ollama
"""

import asyncio
import json
import sqlite3
import os
import logging
from typing import Any, Dict, List, Optional

import aiosqlite
import ollama

# Get logger
logger = logging.getLogger('mcp_server')


class DatabaseManager:
    """Manages SQLite database operations with async support."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use absolute path to ensure database is created in the same directory as the script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(script_dir, "anydb.sqlite")
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with proper settings."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.close()
        logger.info(f"Database initialized at: {self.db_path}")
    
    async def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results."""
        logger.debug(f"Executing query: {query} with params: {params}")
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()
                result = [dict(row) for row in rows]
                logger.debug(f"Query returned {len(result)} rows")
                return result
    
    async def execute_modify(self, query: str, params: tuple = ()) -> int:
        """Execute a modification query (INSERT, UPDATE, DELETE, CREATE)."""
        logger.debug(f"Executing modify: {query} with params: {params}")
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(query, params) as cursor:
                await db.commit()
                rowcount = cursor.rowcount
                logger.debug(f"Modify operation affected {rowcount} rows")
                return rowcount


class OllamaClient:
    """Client for interacting with Ollama for SQL generation."""
    
    def __init__(self, host: str = "localhost", port: int = 1434, model: str = "llama3.1"):
        self.client = ollama.Client(host=f"http://{host}:{port}")
        self.model = model
    
    async def generate_sql(self, instruction: str, table_name: str, schema: Optional[str] = None) -> str:
        """Generate SQL query from natural language instruction."""
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


class DatabaseTools:
    """High-level database tool operations."""
    
    def __init__(self, db_manager: DatabaseManager, ollama_client: OllamaClient):
        self.db_manager = db_manager
        self.ollama_client = ollama_client
    
    async def query_entity(self, entity_name: str, instruction: str = "SELECT all records") -> Dict[str, Any]:
        """Query an entity/table with natural language instructions."""
        logger.info(f"Query Entity - Table: {entity_name}, Instruction: {instruction}")
        
        sql_query = await self.ollama_client.generate_sql(instruction, entity_name)
        results = await self.db_manager.execute_query(sql_query)
        
        response_data = {"sql": sql_query, "results": results}
        logger.info(f"Query returned {len(results)} rows")
        return response_data
    
    async def insert_entity(self, entity_name: str, data: str) -> Dict[str, Any]:
        """Insert records into an entity/table."""
        logger.info(f"Insert Entity - Table: {entity_name}, Data: {data}")
        
        instruction = f"INSERT into {entity_name} with data: {data}"
        sql_query = await self.ollama_client.generate_sql(instruction, entity_name)
        affected_rows = await self.db_manager.execute_modify(sql_query)
        
        response_data = {"sql": sql_query, "affected_rows": affected_rows}
        logger.info(f"Insert affected {affected_rows} rows")
        return response_data
    
    async def update_entity(self, entity_name: str, instruction: str, conditions: str = "") -> Dict[str, Any]:
        """Update records in an entity/table."""
        logger.info(f"Update Entity - Table: {entity_name}, Instruction: {instruction}, Conditions: {conditions}")
        
        full_instruction = f"UPDATE {entity_name} {instruction}"
        if conditions:
            full_instruction += f" WHERE {conditions}"
        
        sql_query = await self.ollama_client.generate_sql(full_instruction, entity_name)
        affected_rows = await self.db_manager.execute_modify(sql_query)
        
        response_data = {"sql": sql_query, "affected_rows": affected_rows}
        logger.info(f"Update affected {affected_rows} rows")
        return response_data
    
    async def delete_entity(self, entity_name: str, conditions: str = "") -> Dict[str, Any]:
        """Delete records from an entity/table."""
        logger.info(f"Delete Entity - Table: {entity_name}, Conditions: {conditions}")
        
        instruction = f"DELETE FROM {entity_name}"
        if conditions:
            instruction += f" WHERE {conditions}"
        
        sql_query = await self.ollama_client.generate_sql(instruction, entity_name)
        affected_rows = await self.db_manager.execute_modify(sql_query)
        
        response_data = {"sql": sql_query, "affected_rows": affected_rows}
        logger.info(f"Delete affected {affected_rows} rows")
        return response_data
    
    async def create_table(self, entity_name: str, schema_description: str) -> Dict[str, Any]:
        """Create a new table with AI-generated schema."""
        logger.info(f"Create Table - Name: {entity_name}, Schema: {schema_description}")
        
        instruction = f"CREATE TABLE {entity_name} with schema: {schema_description}"
        sql_query = await self.ollama_client.generate_sql(instruction, entity_name)
        await self.db_manager.execute_modify(sql_query)
        
        response_data = {
            "sql": sql_query, 
            "message": f"Table {entity_name} created successfully", 
            "database_path": self.db_manager.db_path
        }
        logger.info(f"Table {entity_name} created successfully")
        return response_data
    
    async def execute_raw_query(self, query: str) -> Dict[str, Any]:
        """Execute a raw SQL SELECT query."""
        logger.info(f"Raw SQL Query: {query}")
        
        results = await self.db_manager.execute_query(query)
        logger.info(f"Raw query returned {len(results)} rows")
        return {"query": query, "results": results}
    
    async def execute_raw_modify(self, query: str) -> Dict[str, Any]:
        """Execute a raw SQL modification query."""
        logger.info(f"Raw SQL Modify: {query}")
        
        affected_rows = await self.db_manager.execute_modify(query)
        logger.info(f"Raw modify affected {affected_rows} rows")
        return {"query": query, "affected_rows": affected_rows}