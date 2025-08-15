"""
Tool schema definitions for describing tool interfaces.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from agent_zoo.tools.rate_limit import RateLimit


class ParameterProperty(BaseModel):
    """
    Definition of a single parameter property.

    Represents a parameter in JSON Schema format with common constraints.
    """

    type: str = Field(
        description="JSON Schema type: string, number, integer, boolean, array, object"
    )
    description: str = Field(description="Human-readable description of the parameter")
    default: Any = Field(default=None, description="Default value if not provided")
    enum: list[Any] | None = Field(default=None, description="List of allowed values")
    minimum: float | None = Field(default=None, description="Minimum value (for numbers)")
    maximum: float | None = Field(default=None, description="Maximum value (for numbers)")
    min_length: int | None = Field(default=None, description="Minimum length (for strings/arrays)")
    max_length: int | None = Field(default=None, description="Maximum length (for strings/arrays)")
    pattern: str | None = Field(default=None, description="Regex pattern (for strings)")
    items: dict[str, Any] | None = Field(default=None, description="Schema for array items")
    properties: dict[str, Any] | None = Field(
        default=None, description="Properties for nested objects"
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate that type is a valid JSON Schema type."""
        valid_types = {"string", "number", "integer", "boolean", "array", "object", "null"}
        if v not in valid_types:
            raise ValueError(f"Invalid type '{v}'. Must be one of: {valid_types}")
        return v

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format."""
        return self.model_dump(exclude_none=True)

    def validate_value(self, value: Any) -> tuple[bool, str | None]:
        """
        Validate a value against this property schema.

        Args:
            value: Value to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Type checking
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        expected_type = type_map.get(self.type)
        if expected_type and not isinstance(value, expected_type):
            return False, f"Expected {self.type}, got {type(value).__name__}"

        # Enum validation
        if self.enum is not None and value not in self.enum:
            return False, f"Value must be one of: {self.enum}"

        # Numeric constraints
        if self.type in ("number", "integer"):
            if self.minimum is not None and value < self.minimum:
                return False, f"Value must be >= {self.minimum}"
            if self.maximum is not None and value > self.maximum:
                return False, f"Value must be <= {self.maximum}"

        # String/Array length constraints
        if self.type in ("string", "array"):
            length = len(value)
            if self.min_length is not None and length < self.min_length:
                return False, f"Length must be >= {self.min_length}"
            if self.max_length is not None and length > self.max_length:
                return False, f"Length must be <= {self.max_length}"

        # Pattern matching for strings
        if self.type == "string" and self.pattern:
            import re

            if not re.match(self.pattern, value):
                return False, f"Value must match pattern: {self.pattern}"

        return True, None


class ParameterSchema(BaseModel):
    """
    Structured schema for tool parameters.

    Provides a type-safe way to define tool parameter interfaces
    with automatic validation and JSON Schema generation.
    """

    type: str = Field(default="object", description="Schema type (usually 'object' for functions)")
    properties: dict[str, ParameterProperty | dict[str, Any]] = Field(
        default_factory=dict, description="Parameter definitions"
    )
    required: list[str] = Field(
        default_factory=list, description="List of required parameter names"
    )
    additional_properties: bool = Field(
        default=False, description="Whether to allow additional properties"
    )

    @field_validator("required")
    @classmethod
    def validate_required(cls, v: list[str], info) -> list[str]:
        """Validate that required parameters exist in properties."""
        if "properties" in info.data:
            properties = info.data["properties"]
            for param in v:
                if param not in properties:
                    raise ValueError(f"Required parameter '{param}' not found in properties")
        return v

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to standard JSON Schema format."""
        schema = {
            "type": self.type,
            "properties": {},
            "required": self.required,
            "additionalProperties": self.additional_properties,
        }

        for name, prop in self.properties.items():
            if isinstance(prop, ParameterProperty):
                schema["properties"][name] = prop.to_json_schema()
            else:
                # Allow raw dict for complex schemas
                schema["properties"][name] = prop

        return schema

    def validate_arguments(self, arguments: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate arguments against this schema.

        Args:
            arguments: Arguments to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required parameters
        for param in self.required:
            if param not in arguments:
                errors.append(f"Missing required parameter: {param}")

        # Check each provided argument
        for arg_name, arg_value in arguments.items():
            if arg_name not in self.properties:
                if not self.additional_properties:
                    errors.append(f"Unknown parameter: {arg_name}")
                continue

            prop = self.properties[arg_name]
            if isinstance(prop, ParameterProperty):
                is_valid, error = prop.validate_value(arg_value)
                if not is_valid:
                    errors.append(f"Parameter '{arg_name}': {error}")

        return len(errors) == 0, errors

    @classmethod
    def from_dict(cls, schema_dict: dict[str, Any]) -> "ParameterSchema":
        """
        Create ParameterSchema from JSON Schema dict.

        Args:
            schema_dict: JSON Schema dictionary

        Returns:
            ParameterSchema instance
        """
        properties = {}
        for name, prop_dict in schema_dict.get("properties", {}).items():
            if isinstance(prop_dict, dict) and "type" in prop_dict:
                # Try to create ParameterProperty
                try:
                    properties[name] = ParameterProperty(**prop_dict)
                except:
                    # Fallback to raw dict for complex schemas
                    properties[name] = prop_dict
            else:
                properties[name] = prop_dict

        return cls(
            type=schema_dict.get("type", "object"),
            properties=properties,
            required=schema_dict.get("required", []),
            additional_properties=schema_dict.get("additionalProperties", False),
        )


class ToolExample(BaseModel):
    """Example usage of a tool."""

    description: str = Field(description="What this example demonstrates")
    arguments: dict[str, Any] = Field(description="Example input arguments")
    expected_output: Any = Field(description="Expected output")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


class ToolSchema(BaseModel):
    """
    Schema definition for a tool interface.

    This defines the contract that a tool exposes to agents,
    similar to how AgentConfig defines configuration for agents.
    """

    # Identification
    name: str = Field(description="Unique tool identifier")
    description: str = Field(description="Human-readable description of what the tool does")

    # Interface definition with structured or raw schema
    parameters: ParameterSchema | dict[str, Any] = Field(
        default_factory=ParameterSchema,
        description="Parameter schema (structured or raw JSON Schema)",
    )
    returns: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for return value"
    )

    # Documentation
    examples: list[ToolExample] = Field(default_factory=list, description="Usage examples")

    # Categorization and constraints
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    requires_auth: bool = Field(default=False, description="Whether authentication is required")
    rate_limits: list[RateLimit] = Field(
        default_factory=list, description="Rate limiting strategies to apply"
    )
    timeout_seconds: float = Field(default=30.0, description="Execution timeout in seconds")

    # Capabilities
    supports_streaming: bool = Field(
        default=False, description="Whether the tool supports streaming responses"
    )
    is_stateful: bool = Field(
        default=False, description="Whether the tool maintains state between calls"
    )

    def to_openai_function(self) -> dict[str, Any]:
        """
        Convert to OpenAI function calling format.

        Returns:
            Dictionary in OpenAI function format
        """
        params = self.parameters
        if isinstance(params, ParameterSchema):
            params = params.to_json_schema()

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }

    def to_anthropic_tool(self) -> dict[str, Any]:
        """
        Convert to Anthropic tool use format.

        Returns:
            Dictionary in Anthropic tool format
        """
        params = self.parameters
        if isinstance(params, ParameterSchema):
            params = params.to_json_schema()

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": params,
        }

    def to_langchain_tool(self) -> dict[str, Any]:
        """
        Convert to LangChain tool format.

        Returns:
            Dictionary in LangChain tool format
        """
        params = self.parameters
        if isinstance(params, ParameterSchema):
            params = params.to_json_schema()

        return {
            "name": self.name,
            "description": self.description,
            "args_schema": params,
            "return_direct": False,
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate arguments against the schema.

        Args:
            arguments: Arguments to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        if isinstance(self.parameters, ParameterSchema):
            # Use structured validation
            return self.parameters.validate_arguments(arguments)
        else:
            # Fallback for raw dict schemas
            if not isinstance(self.parameters, dict):
                return False, ["Invalid parameter schema type"]

            errors = []

            # Basic validation for dict-based schemas
            required = self.parameters.get("required", [])
            for param in required:
                if param not in arguments:
                    errors.append(f"Missing required parameter: {param}")

            # Check if properties exist (defensive)
            properties = self.parameters.get("properties", {})
            if properties:
                for arg_name in arguments:
                    if arg_name not in properties:
                        if not self.parameters.get("additionalProperties", False):
                            errors.append(f"Unknown parameter: {arg_name}")

            return len(errors) == 0, errors

    def get_required_parameters(self) -> list[str]:
        """Get list of required parameters."""
        if isinstance(self.parameters, ParameterSchema):
            return self.parameters.required
        elif isinstance(self.parameters, dict):
            return self.parameters.get("required", [])
        return []

    def get_optional_parameters(self) -> list[str]:
        """Get list of optional parameters."""
        if isinstance(self.parameters, ParameterSchema):
            all_params = list(self.parameters.properties.keys())
        elif isinstance(self.parameters, dict):
            all_params = list(self.parameters.get("properties", {}).keys())
        else:
            return []

        required = set(self.get_required_parameters())
        return [p for p in all_params if p not in required]

    def __repr__(self) -> str:
        """String representation."""
        return f"ToolSchema(name={self.name}, tags={self.tags})"
