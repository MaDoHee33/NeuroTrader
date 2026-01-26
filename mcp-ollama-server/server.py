#!/usr/bin/env python3
"""
MCP Server for Ollama Integration
Provides tools to interact with Ollama models (e.g., DeepSeek) via MCP protocol.
"""

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from ollama_client import OllamaClient

# Default model to use (change this to match your Ollama models)
DEFAULT_MODEL = "qwen3-coder:480b-cloud"

# Initialize MCP Server
server = Server("ollama-server")
ollama = OllamaClient()


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="ask_deepseek",
            description="Ask a question to DeepSeek AI model via Ollama. Use this for general questions, code help, or analysis.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question or prompt to send to DeepSeek"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt to set the context/role"
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="ask_with_context",
            description="Ask DeepSeek a question with additional context. Useful for code review, document analysis, etc.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "The context (code, document, etc.) to analyze"
                    },
                    "question": {
                        "type": "string",
                        "description": "The question about the context"
                    }
                },
                "required": ["context", "question"]
            }
        ),
        Tool(
            name="list_ollama_models",
            description="List all available models in Ollama server",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="check_ollama_status",
            description="Check if Ollama server is running and responsive",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    
    if name == "ask_deepseek":
        question = arguments.get("question", "")
        system_prompt = arguments.get("system_prompt")
        
        try:
            response = await ollama.generate(
                model=DEFAULT_MODEL,
                prompt=question,
                system=system_prompt
            )
            return [TextContent(type="text", text=response)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "ask_with_context":
        context = arguments.get("context", "")
        question = arguments.get("question", "")
        
        prompt = f"""Context:
```
{context}
```

Question: {question}"""
        
        try:
            response = await ollama.generate(
                model=DEFAULT_MODEL,
                prompt=prompt
            )
            return [TextContent(type="text", text=response)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "list_ollama_models":
        try:
            models = await ollama.list_models()
            if not models:
                return [TextContent(type="text", text="No models found in Ollama")]
            
            result = "Available models:\n"
            for model in models:
                name_str = model.get("name", "unknown")
                size = model.get("size", 0)
                size_gb = size / (1024 ** 3) if size else 0
                result += f"- {name_str} ({size_gb:.1f} GB)\n"
            
            return [TextContent(type="text", text=result)]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "check_ollama_status":
        try:
            is_running = await ollama.is_server_running()
            if is_running:
                return [TextContent(type="text", text="✅ Ollama server is running and responsive")]
            else:
                return [TextContent(type="text", text="❌ Ollama server is not responding")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error checking status: {str(e)}")]
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
