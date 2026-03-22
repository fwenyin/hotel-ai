"""Base tool interface for agent tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type


class BaseTool(ABC):
    """Abstract base class for agent tools."""

    def __init__(self, config: Dict):
        """Initialize tool with database path.

        Args:
            config: Configuration dictionary
        """
        self.config = config

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool's primary function.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Tool-specific output
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get tool description."""
        pass

    @property
    def input_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool inputs.

        Returns:
            Dictionary describing expected inputs
        """
        return {}


class ToolRegistry:
    """Registry for dynamic tool discovery and registration."""

    _tools: Dict[str, type] = {}

    @classmethod
    def register(cls, tool_class: Type["BaseTool"]) -> Type["BaseTool"]:
        """Register a tool class.

        Args:
            tool_class: Tool class to register

        Returns:
            The tool class (for decorator chaining)
        """
        cls._tools[tool_class.__name__] = tool_class
        return tool_class

    @classmethod
    def instantiate_all(cls, config: Dict) -> Dict[str, BaseTool]:
        """Instantiate all registered tools.

        Args:
            config: Configuration dictionary

        Returns:
            Dictionary mapping tool instance names to instances
        """
        instances = {}
        for tool_class in cls._tools.values():
            instance = tool_class(config)
            instances[instance.name] = instance
        return instances
