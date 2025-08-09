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
import re
from typing import Any, Dict, List, Optional, Set, Tuple

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
    
    async def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get the schema information for a table."""
        logger.debug(f"Getting schema for table: {table_name}")
        query = f"PRAGMA table_info({table_name})"
        return await self.execute_query(query)
    
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = await self.execute_query(query, (table_name,))
        return len(result) > 0


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
    
    def _extract_columns_from_data(self, data_instruction: str) -> Set[str]:
        """Extract potential column names from user data instruction using pattern matching."""
        # Convert to lowercase for matching
        data_lower = data_instruction.lower()
        
        # Common column indicators and their patterns
        column_patterns = [
            # Direct indicators: "with isbn:", "with format:", etc.
            r'with\s+(\w+):',
            r'with\s+(\w+)\s+',
            # Value assignments: "isbn is", "format is", etc.
            r'(\w+)\s+is\s+',
            # Preposition patterns: "in paperback format", "available in format"
            r'in\s+\w+\s+(\w+)',
            r'available\s+in\s+(\w+)',
            r'released\s+on\s+[\w\s]+\s+(date|time)',
            # Common field patterns with better context
            r'(\w*name)\b',
            r'(\w*date)\b',
            r'(\w*time)\b',
            r'(\w*id)\b',
            r'(\w*number)\b',
            r'(\w*code)\b',
            r'(\w*format)\b',
            r'(\w*type)\b',
            r'(\w*status)\b',
            r'(\w*price)\b',
            r'(\w*category)\b',
            r'(\w*description)\b',
            # Specific patterns for common scenarios
            r'by\s+[\w\s]+\s+(author|writer)',
            r'released\s+[\w\s]+\s+(release.*date|publication.*date)',
            r'available\s+[\w\s]+\s+(format|edition)',
        ]
        
        extracted_columns = set()
        
        for pattern in column_patterns:
            matches = re.findall(pattern, data_lower)
            extracted_columns.update(matches)
        
        # Add specific mappings for common phrases
        phrase_mappings = {
            'by': 'author',
            'released on': 'release_date',
            'published on': 'publication_date',
            'available in': 'format',
        }
        
        for phrase, column in phrase_mappings.items():
            if phrase in data_lower:
                extracted_columns.add(column)
        
        # Remove common words that aren't likely columns
        stop_words = {'the', 'and', 'or', 'is', 'are', 'was', 'were', 'with', 'in', 'on', 'at', 'by', 'for', 'to', 'of', 'a', 'an', 'new', 'old', 'first', 'last'}
        extracted_columns = {col for col in extracted_columns if col not in stop_words and len(col) > 1}
        
        logger.debug(f"Extracted potential columns: {extracted_columns}")
        return extracted_columns
    
    async def _get_missing_columns(self, table_name: str, potential_columns: Set[str]) -> Set[str]:
        """Get columns that don't exist in the current table schema."""
        if not await self.db_manager.table_exists(table_name):
            return potential_columns
        
        schema = await self.db_manager.get_table_schema(table_name)
        existing_columns = {col['name'].lower() for col in schema}
        
        missing_columns = potential_columns - existing_columns
        logger.debug(f"Missing columns for {table_name}: {missing_columns}")
        return missing_columns
    
    async def _generate_column_definitions(self, table_name: str, column_names: Set[str], context_data: str) -> str:
        """Use Ollama to generate appropriate column definitions based on context."""
        if not column_names:
            return ""
        
        prompt = f"""
        Based on this data context: "{context_data}"
        Generate SQL ALTER TABLE statements to add these columns to table '{table_name}': {', '.join(column_names)}
        
        Consider the context to determine appropriate data types (TEXT, INTEGER, REAL, DATE, etc.).
        Return only the ALTER TABLE statements, one per line, no explanations.
        Example format: ALTER TABLE {table_name} ADD COLUMN column_name TEXT;
        """
        
        try:
            response = self.ollama_client.client.chat(
                model=self.ollama_client.model,
                messages=[{"role": "user", "content": prompt}]
            )
            alter_statements = response["message"]["content"].strip()
            logger.debug(f"Generated ALTER statements: {alter_statements}")
            return alter_statements
        except Exception as e:
            logger.error(f"Error generating column definitions: {str(e)}")
            # Fallback: create TEXT columns
            fallback_statements = []
            for col in column_names:
                fallback_statements.append(f"ALTER TABLE {table_name} ADD COLUMN {col} TEXT;")
            return "\n".join(fallback_statements)
    
    async def _auto_modify_table_schema(self, table_name: str, data_instruction: str) -> List[str]:
        """Automatically modify table schema based on data instruction."""
        logger.info(f"Auto-modifying schema for {table_name} based on: {data_instruction}")
        
        # Extract potential columns from the instruction
        potential_columns = self._extract_columns_from_data(data_instruction)
        
        if not potential_columns:
            logger.debug("No potential new columns detected")
            return []
        
        # Get missing columns
        missing_columns = await self._get_missing_columns(table_name, potential_columns)
        
        if not missing_columns:
            logger.debug("No missing columns to add")
            return []
        
        # Generate ALTER TABLE statements
        alter_statements = await self._generate_column_definitions(table_name, missing_columns, data_instruction)
        
        if not alter_statements:
            return []
        
        # Execute the ALTER statements
        executed_statements = []
        for statement in alter_statements.split('\n'):
            statement = statement.strip()
            if statement and statement.upper().startswith('ALTER TABLE'):
                try:
                    await self.db_manager.execute_modify(statement)
                    executed_statements.append(statement)
                    logger.info(f"Executed: {statement}")
                except Exception as e:
                    logger.error(f"Failed to execute: {statement}, Error: {str(e)}")
        
        return executed_statements
    
    async def query_entity(self, entity_name: str, instruction: str = "SELECT all records") -> Dict[str, Any]:
        """Query an entity/table with natural language instructions."""
        logger.info(f"Query Entity - Table: {entity_name}, Instruction: {instruction}")
        
        sql_query = await self.ollama_client.generate_sql(instruction, entity_name)
        results = await self.db_manager.execute_query(sql_query)
        
        response_data = {"sql": sql_query, "results": results}
        logger.info(f"Query returned {len(results)} rows")
        return response_data
    
    async def insert_entity(self, entity_name: str, data: str) -> Dict[str, Any]:
        """Insert records into an entity/table with automatic schema modification."""
        logger.info(f"Insert Entity - Table: {entity_name}, Data: {data}")
        
        # Automatically modify table schema if needed
        schema_modifications = await self._auto_modify_table_schema(entity_name, data)
        
        instruction = f"INSERT into {entity_name} with data: {data}"
        sql_query = await self.ollama_client.generate_sql(instruction, entity_name)
        affected_rows = await self.db_manager.execute_modify(sql_query)
        
        response_data = {
            "sql": sql_query, 
            "affected_rows": affected_rows,
            "schema_modifications": schema_modifications
        }
        
        if schema_modifications:
            response_data["message"] = f"Table schema automatically updated with {len(schema_modifications)} new columns"
        
        logger.info(f"Insert affected {affected_rows} rows")
        return response_data
    
    async def update_entity(self, entity_name: str, instruction: str, conditions: str = "") -> Dict[str, Any]:
        """Update records in an entity/table with automatic schema modification."""
        logger.info(f"Update Entity - Table: {entity_name}, Instruction: {instruction}, Conditions: {conditions}")
        
        # Check if the instruction contains new column data
        combined_instruction = f"{instruction} {conditions}"
        schema_modifications = await self._auto_modify_table_schema(entity_name, combined_instruction)
        
        full_instruction = f"UPDATE {entity_name} {instruction}"
        if conditions:
            full_instruction += f" WHERE {conditions}"
        
        sql_query = await self.ollama_client.generate_sql(full_instruction, entity_name)
        affected_rows = await self.db_manager.execute_modify(sql_query)
        
        response_data = {
            "sql": sql_query, 
            "affected_rows": affected_rows,
            "schema_modifications": schema_modifications
        }
        
        if schema_modifications:
            response_data["message"] = f"Table schema automatically updated with {len(schema_modifications)} new columns"
        
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