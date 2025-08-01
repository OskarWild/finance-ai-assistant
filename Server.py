from dotenv import load_dotenv
from fastmcp import FastMCP
import json
import os

load_dotenv()

mcp = FastMCP("Employees MCP HTTP Agent")

@mcp.tool(description="Получение списка всех сотрудников с описанием их позиции и обьязанностей", name="all_employee_details")
def summarize_report() -> str:
    """Получение всех сотрудников с описанием их позиции и обьязанностей"""
    
if __name__ == "__main__":
    mcp.run(transport="streamable-http", 
            host="127.0.0.1", 
            port=9007 
            )