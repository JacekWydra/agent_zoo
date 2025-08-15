"""
Unit tests for tool schema definitions.
"""

import pytest
from pydantic import ValidationError

from agent_zoo.tools.rate_limit import CallRateLimit
from agent_zoo.tools.schema import (
    ParameterProperty,
    ParameterSchema,
    ToolExample,
    ToolSchema,
)


class TestParameterProperty:
    """Tests for ParameterProperty."""

    def test_basic_string_property(self):
        """Test basic string parameter property."""
        prop = ParameterProperty(
            type="string",
            description="A string parameter",
        )

        assert prop.type == "string"
        assert prop.description == "A string parameter"
        assert prop.default is None
        assert prop.enum is None

    def test_number_property_with_constraints(self):
        """Test number property with min/max constraints."""
        prop = ParameterProperty(
            type="number",
            description="A number between 0 and 100",
            minimum=0,
            maximum=100,
        )

        assert prop.type == "number"
        assert prop.minimum == 0
        assert prop.maximum == 100

    def test_enum_property(self):
        """Test property with enum values."""
        prop = ParameterProperty(
            type="string",
            description="Operation type",
            enum=["add", "subtract", "multiply", "divide"],
        )

        assert prop.enum == ["add", "subtract", "multiply", "divide"]

    def test_array_property(self):
        """Test array property."""
        prop = ParameterProperty(
            type="array",
            description="List of items",
            min_length=1,
            max_length=10,
            items={"type": "string"},
        )

        assert prop.type == "array"
        assert prop.min_length == 1
        assert prop.max_length == 10
        assert prop.items == {"type": "string"}

    def test_object_property(self):
        """Test object property with nested properties."""
        prop = ParameterProperty(
            type="object",
            description="Complex object",
            properties={
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        )

        assert prop.type == "object"
        assert prop.properties["name"]["type"] == "string"

    def test_string_with_pattern(self):
        """Test string property with regex pattern."""
        prop = ParameterProperty(
            type="string",
            description="Email address",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        )

        assert prop.pattern == r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    def test_invalid_type(self):
        """Test validation of invalid type."""
        with pytest.raises(ValidationError):
            ParameterProperty(
                type="invalid_type",
                description="Test",
            )

    def test_to_json_schema(self):
        """Test conversion to JSON Schema."""
        prop = ParameterProperty(
            type="integer",
            description="Age",
            minimum=0,
            maximum=150,
            default=25,
        )

        schema = prop.to_json_schema()

        assert schema["type"] == "integer"
        assert schema["description"] == "Age"
        assert schema["minimum"] == 0
        assert schema["maximum"] == 150
        assert schema["default"] == 25

    def test_validate_value_type_checking(self):
        """Test value validation for type checking."""
        # String property
        str_prop = ParameterProperty(type="string", description="Test")
        assert str_prop.validate_value("hello")[0] is True
        assert str_prop.validate_value(123)[0] is False

        # Number property
        num_prop = ParameterProperty(type="number", description="Test")
        assert num_prop.validate_value(42.5)[0] is True
        assert num_prop.validate_value("not a number")[0] is False

        # Boolean property
        bool_prop = ParameterProperty(type="boolean", description="Test")
        assert bool_prop.validate_value(True)[0] is True
        assert bool_prop.validate_value("true")[0] is False

    def test_validate_value_enum(self):
        """Test value validation for enum constraints."""
        prop = ParameterProperty(
            type="string",
            description="Color",
            enum=["red", "green", "blue"],
        )

        assert prop.validate_value("red")[0] is True
        assert prop.validate_value("yellow")[0] is False
        assert prop.validate_value("yellow")[1] == "Value must be one of: ['red', 'green', 'blue']"

    def test_validate_value_numeric_constraints(self):
        """Test value validation for numeric constraints."""
        prop = ParameterProperty(
            type="integer",
            description="Score",
            minimum=0,
            maximum=100,
        )

        assert prop.validate_value(50)[0] is True
        assert prop.validate_value(-1)[0] is False
        assert prop.validate_value(101)[0] is False

    def test_validate_value_string_length(self):
        """Test value validation for string length constraints."""
        prop = ParameterProperty(
            type="string",
            description="Username",
            min_length=3,
            max_length=20,
        )

        assert prop.validate_value("john")[0] is True
        assert prop.validate_value("ab")[0] is False
        assert prop.validate_value("a" * 21)[0] is False

    def test_validate_value_pattern(self):
        """Test value validation for regex pattern."""
        prop = ParameterProperty(
            type="string",
            description="Phone",
            pattern=r"^\d{3}-\d{3}-\d{4}$",
        )

        assert prop.validate_value("123-456-7890")[0] is True
        assert prop.validate_value("1234567890")[0] is False


class TestParameterSchema:
    """Tests for ParameterSchema."""

    def test_basic_schema(self):
        """Test basic parameter schema."""
        schema = ParameterSchema(
            properties={
                "name": ParameterProperty(
                    type="string",
                    description="Name",
                ),
                "age": ParameterProperty(
                    type="integer",
                    description="Age",
                    minimum=0,
                ),
            },
            required=["name"],
        )

        assert "name" in schema.properties
        assert "age" in schema.properties
        assert schema.required == ["name"]
        assert schema.additional_properties is False

    def test_schema_with_raw_dict_properties(self):
        """Test schema with raw dict properties."""
        schema = ParameterSchema(
            properties={
                "complex": {
                    "type": "object",
                    "properties": {
                        "nested": {"type": "string"},
                    },
                },
            },
        )

        assert "complex" in schema.properties
        assert isinstance(schema.properties["complex"], dict)

    def test_required_validation(self):
        """Test validation of required parameters."""
        # Should fail - required parameter doesn't exist in properties
        with pytest.raises(ValidationError):
            ParameterSchema(
                properties={"name": ParameterProperty(type="string", description="Name")},
                required=["name", "nonexistent"],
            )

    def test_to_json_schema(self):
        """Test conversion to JSON Schema."""
        schema = ParameterSchema(
            properties={
                "query": ParameterProperty(
                    type="string",
                    description="Search query",
                ),
                "limit": ParameterProperty(
                    type="integer",
                    description="Result limit",
                    default=10,
                ),
            },
            required=["query"],
            additional_properties=True,
        )

        json_schema = schema.to_json_schema()

        assert json_schema["type"] == "object"
        assert "query" in json_schema["properties"]
        assert json_schema["required"] == ["query"]
        assert json_schema["additionalProperties"] is True

    def test_validate_arguments(self):
        """Test argument validation."""
        schema = ParameterSchema(
            properties={
                "operation": ParameterProperty(
                    type="string",
                    enum=["add", "subtract"],
                    description="Operation",
                ),
                "a": ParameterProperty(type="number", description="First number"),
                "b": ParameterProperty(type="number", description="Second number"),
            },
            required=["operation", "a", "b"],
        )

        # Valid arguments
        is_valid, errors = schema.validate_arguments({
            "operation": "add",
            "a": 5,
            "b": 3,
        })
        assert is_valid is True
        assert errors == []

        # Missing required parameter
        is_valid, errors = schema.validate_arguments({
            "operation": "add",
            "a": 5,
        })
        assert is_valid is False
        assert "Missing required parameter: b" in errors

        # Invalid enum value
        is_valid, errors = schema.validate_arguments({
            "operation": "multiply",
            "a": 5,
            "b": 3,
        })
        assert is_valid is False
        assert any("operation" in error for error in errors)

        # Unknown parameter (not allowed)
        is_valid, errors = schema.validate_arguments({
            "operation": "add",
            "a": 5,
            "b": 3,
            "c": 10,
        })
        assert is_valid is False
        assert "Unknown parameter: c" in errors

    def test_from_dict(self):
        """Test creating schema from dictionary."""
        schema_dict = {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Input text",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum length",
                    "minimum": 1,
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        }

        schema = ParameterSchema.from_dict(schema_dict)

        assert "text" in schema.properties
        assert isinstance(schema.properties["text"], ParameterProperty)
        assert schema.properties["text"].type == "string"
        assert schema.required == ["text"]


class TestToolExample:
    """Tests for ToolExample."""

    def test_tool_example(self):
        """Test tool example creation."""
        example = ToolExample(
            description="Add two numbers",
            arguments={"operation": "add", "a": 5, "b": 3},
            expected_output=8,
        )

        assert example.description == "Add two numbers"
        assert example.arguments["operation"] == "add"
        assert example.expected_output == 8

    def test_example_to_dict(self):
        """Test example conversion to dict."""
        example = ToolExample(
            description="Search example",
            arguments={"query": "Python"},
            expected_output=["result1", "result2"],
        )

        example_dict = example.to_dict()

        assert example_dict["description"] == "Search example"
        assert example_dict["arguments"]["query"] == "Python"
        assert example_dict["expected_output"] == ["result1", "result2"]


class TestToolSchema:
    """Tests for ToolSchema."""

    def test_basic_tool_schema(self):
        """Test basic tool schema creation."""
        schema = ToolSchema(
            name="calculator",
            description="Performs calculations",
        )

        assert schema.name == "calculator"
        assert schema.description == "Performs calculations"
        assert isinstance(schema.parameters, ParameterSchema)
        assert schema.tags == []
        assert schema.rate_limits == []
        assert schema.timeout_seconds == 30.0

    def test_tool_schema_with_parameters(self):
        """Test tool schema with structured parameters."""
        params = ParameterSchema(
            properties={
                "text": ParameterProperty(
                    type="string",
                    description="Input text",
                ),
            },
            required=["text"],
        )

        schema = ToolSchema(
            name="text_processor",
            description="Process text",
            parameters=params,
            tags=["text", "nlp"],
        )

        assert schema.name == "text_processor"
        assert isinstance(schema.parameters, ParameterSchema)
        assert schema.tags == ["text", "nlp"]

    def test_tool_schema_with_dict_parameters(self):
        """Test tool schema with raw dict parameters."""
        schema = ToolSchema(
            name="api_caller",
            description="Call API",
            parameters={
                "type": "object",
                "properties": {
                    "endpoint": {"type": "string"},
                },
                "required": ["endpoint"],
            },
        )

        assert isinstance(schema.parameters, ParameterSchema)
        assert schema.parameters.properties["endpoint"]["type"] == "string"

    def test_tool_schema_with_rate_limits(self):
        """Test tool schema with rate limits."""
        rate_limit = CallRateLimit(max_calls=10, window_seconds=60)

        schema = ToolSchema(
            name="limited_tool",
            description="Rate limited tool",
            rate_limits=[rate_limit],
        )

        assert len(schema.rate_limits) == 1
        assert schema.rate_limits[0].max_calls == 10

    def test_tool_schema_with_examples(self):
        """Test tool schema with examples."""
        example = ToolExample(
            description="Example usage",
            arguments={"input": "test"},
            expected_output="result",
        )

        schema = ToolSchema(
            name="example_tool",
            description="Tool with examples",
            examples=[example],
        )

        assert len(schema.examples) == 1
        assert schema.examples[0].description == "Example usage"

    def test_to_openai_function(self, simple_tool_schema):
        """Test conversion to OpenAI function format."""
        openai_func = simple_tool_schema.to_openai_function()

        assert openai_func["type"] == "function"
        assert openai_func["function"]["name"] == "calculator"
        assert openai_func["function"]["description"] == "Performs basic arithmetic operations"
        assert "properties" in openai_func["function"]["parameters"]

    def test_to_anthropic_tool(self, simple_tool_schema):
        """Test conversion to Anthropic tool format."""
        anthropic_tool = simple_tool_schema.to_anthropic_tool()

        assert anthropic_tool["name"] == "calculator"
        assert anthropic_tool["description"] == "Performs basic arithmetic operations"
        assert "input_schema" in anthropic_tool

    def test_to_langchain_tool(self, simple_tool_schema):
        """Test conversion to LangChain tool format."""
        langchain_tool = simple_tool_schema.to_langchain_tool()

        assert langchain_tool["name"] == "calculator"
        assert langchain_tool["description"] == "Performs basic arithmetic operations"
        assert "args_schema" in langchain_tool
        assert langchain_tool["return_direct"] is False

    def test_validate_arguments_with_structured_params(self, simple_tool_schema):
        """Test argument validation with structured parameters."""
        # Valid arguments
        is_valid, errors = simple_tool_schema.validate_arguments({
            "operation": "add",
            "a": 5,
            "b": 3,
        })
        assert is_valid is True

        # Invalid arguments
        is_valid, errors = simple_tool_schema.validate_arguments({
            "operation": "invalid",
            "a": 5,
        })
        assert is_valid is False
        assert len(errors) > 0

    def test_validate_arguments_with_dict_params(self):
        """Test argument validation with dict parameters."""
        schema = ToolSchema(
            name="test",
            description="Test",
            parameters={
                "properties": {"x": {"type": "number"}},
                "required": ["x"],
            },
        )

        is_valid, errors = schema.validate_arguments({"x": 42})
        assert is_valid is True

        is_valid, errors = schema.validate_arguments({})
        assert is_valid is False
        assert "Missing required parameter: x" in errors

    def test_get_required_parameters(self, simple_tool_schema):
        """Test getting required parameters."""
        required = simple_tool_schema.get_required_parameters()
        assert set(required) == {"operation", "a", "b"}

    def test_get_optional_parameters(self):
        """Test getting optional parameters."""
        schema = ToolSchema(
            name="test",
            description="Test",
            parameters=ParameterSchema(
                properties={
                    "required_param": ParameterProperty(type="string", description="Required"),
                    "optional_param": ParameterProperty(type="string", description="Optional"),
                },
                required=["required_param"],
            ),
        )

        optional = schema.get_optional_parameters()
        assert optional == ["optional_param"]

    def test_tool_capabilities(self):
        """Test tool capability flags."""
        # Since supports_streaming and is_stateful don't exist in ToolSchema,
        # test other existing capabilities instead
        schema = ToolSchema(
            name="stream_tool",
            description="Streaming tool",
            requires_auth=True,
            timeout_seconds=60.0,
        )

        assert schema.requires_auth is True
        assert schema.timeout_seconds == 60.0
