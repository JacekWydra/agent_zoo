"""
Tool registry for managing and discovering available tools.
"""

from typing import Any, Callable
import structlog

from agent_zoo.tools.tool import Tool
from agent_zoo.tools.schema import ToolSchema

logger = structlog.get_logger()


class ToolRegistry:
    """
    Registry for managing available tools.
    
    Provides centralized tool management including registration,
    discovery, and schema access for agents.
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self._tools: dict[str, Tool] = {}
        self._categories: dict[str, list[str]] = {}
        logger.info("Tool registry initialized")
    
    def register(self, tool: Tool) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If a tool with the same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        
        self._tools[tool.name] = tool
        
        # Update categories based on tags
        for tag in tool.schema.tags:
            if tag not in self._categories:
                self._categories[tag] = []
            self._categories[tag].append(tool.name)
        
        logger.info(
            f"Registered tool: {tool.name}",
            tags=tool.schema.tags,
            is_async=tool.is_async
        )
    
    def register_function(
        self,
        schema: ToolSchema,
        function: Callable
    ) -> Tool:
        """
        Helper to register a function as a tool.
        
        Args:
            schema: Tool schema
            function: Function to register
            
        Returns:
            Created and registered Tool instance
        """
        tool = Tool(schema=schema, function=function)
        self.register(tool)
        return tool
    
    def unregister(self, name: str) -> bool:
        """
        Remove a tool from the registry.
        
        Args:
            name: Name of the tool to remove
            
        Returns:
            True if tool was removed, False if not found
        """
        if name not in self._tools:
            return False
        
        tool = self._tools[name]
        
        # Remove from categories
        for tag in tool.schema.tags:
            if tag in self._categories:
                self._categories[tag].remove(name)
                if not self._categories[tag]:
                    del self._categories[tag]
        
        # Remove tool
        del self._tools[name]
        logger.info(f"Unregistered tool: {name}")
        return True
    
    def get(self, name: str) -> Tool | None:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def get_all(self) -> list[Tool]:
        """
        Get all registered tools.
        
        Returns:
            List of all tools
        """
        return list(self._tools.values())
    
    def get_schemas(self) -> list[ToolSchema]:
        """
        Get schemas for all registered tools.
        
        Returns:
            List of tool schemas
        """
        return [tool.schema for tool in self._tools.values()]
    
    def get_by_tag(self, tag: str) -> list[Tool]:
        """
        Get tools by tag/category.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of tools with the given tag
        """
        tool_names = self._categories.get(tag, [])
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def get_tags(self) -> list[str]:
        """
        Get all available tags.
        
        Returns:
            List of all tags
        """
        return list(self._categories.keys())
    
    def search(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        requires_auth: bool | None = None
    ) -> list[Tool]:
        """
        Search for tools based on criteria.
        
        Args:
            query: Text to search in name and description
            tags: Tags to filter by (matches any)
            requires_auth: Filter by authentication requirement
            
        Returns:
            List of matching tools
        """
        results = list(self._tools.values())
        
        # Filter by query
        if query:
            query_lower = query.lower()
            results = [
                tool for tool in results
                if (query_lower in tool.name.lower() or 
                    query_lower in tool.description.lower())
            ]
        
        # Filter by tags
        if tags:
            results = [
                tool for tool in results
                if any(tag in tool.schema.tags for tag in tags)
            ]
        
        # Filter by auth requirement
        if requires_auth is not None:
            results = [
                tool for tool in results
                if tool.schema.requires_auth == requires_auth
            ]
        
        return results
    
    def get_openai_functions(self) -> list[dict[str, Any]]:
        """
        Get all tools in OpenAI function format.
        
        Returns:
            List of OpenAI function definitions
        """
        return [tool.schema.to_openai_function() for tool in self._tools.values()]
    
    def get_anthropic_tools(self) -> list[dict[str, Any]]:
        """
        Get all tools in Anthropic tool format.
        
        Returns:
            List of Anthropic tool definitions
        """
        return [tool.schema.to_anthropic_tool() for tool in self._tools.values()]
    
    def get_langchain_tools(self) -> list[dict[str, Any]]:
        """
        Get all tools in LangChain format.
        
        Returns:
            List of LangChain tool definitions
        """
        return [tool.schema.to_langchain_tool() for tool in self._tools.values()]
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
        self._categories.clear()
        logger.info("Tool registry cleared")
    
    def get_metrics(self) -> dict[str, Any]:
        """
        Get metrics for all registered tools.
        
        Returns:
            Dictionary with tool metrics
        """
        return {
            "total_tools": len(self._tools),
            "categories": len(self._categories),
            "tools_by_category": {
                tag: len(tools) for tag, tools in self._categories.items()
            },
            "tool_metrics": {
                name: tool.get_metrics() for name, tool in self._tools.items()
            }
        }
    
    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ToolRegistry(tools={len(self._tools)}, categories={len(self._categories)})"