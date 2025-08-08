#!/usr/bin/env python3
"""
MCP Server stdio entry point for Claude Desktop integration.

This is a clean entry point that produces no console output,
only MCP protocol messages via stdio.
"""

import os
import sys
import asyncio

# Disable all startup messages for clean stdio communication
os.environ['MCP_DISABLE_STARTUP_MESSAGES'] = '1'

# Add current directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server import main as mcp_main

if __name__ == "__main__":
    try:
        asyncio.run(mcp_main())
    except Exception:
        # Exit silently - all errors are logged to mcp_server.log
        sys.exit(1)