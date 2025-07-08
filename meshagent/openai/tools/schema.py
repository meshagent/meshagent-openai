def validate_response_format(response_format) -> str | None:
    """
    Validates a response format according to the OpenAI Structured Outputs specification.

    See https://platform.openai.com/docs/guides/structured-outputs for details.

    Note: This code is up to date as of January 21, 2024
    """

    # Check that response_format is a dictionary
    if not isinstance(response_format, dict):
        return "Error: Response format must be a dictionary."

    # Check that response_format contains exactly "type" and "json_schema" keys
    if set(response_format.keys()) != {"type", "json_schema"}:
        return (
            "Error: Response format must contain exactly 'type' and 'json_schema' keys."
        )

    # Check that response format has type=json_schema
    if "type" not in response_format or response_format["type"] != "json_schema":
        return "Error: Response format must have type 'json_schema'."

    # Check that the "json_schema" is a dict
    if "json_schema" not in response_format or not isinstance(
        response_format["json_schema"], dict
    ):
        return "Error: 'json_schema' key must be a dictionary."

    # Check that "json_schema" contains exactly "name" and "schema" keys, and optionally "description" and "strict" keys
    required_keys = {"name", "schema"}
    optional_keys = {"description", "strict"}
    if set(response_format["json_schema"].keys()) != required_keys.union(optional_keys):
        return "Error: 'json_schema' key must contain exactly 'name', 'schema', and optionally 'description' and 'strict' keys."

    # Check that "json_schema" contains a "name" string
    if "name" not in response_format["json_schema"] or not isinstance(
        response_format["json_schema"]["name"], str
    ):
        return "Error: 'name' key must be a string."

    # Check that "description" is a string if present
    if "description" in response_format["json_schema"] and not isinstance(
        response_format["json_schema"]["description"], str
    ):
        return "Error: 'description' key must be a string."

    # Check that "json_schema" contains a "schema" dict
    if "schema" not in response_format["json_schema"] or not isinstance(
        response_format["json_schema"]["schema"], dict
    ):
        return "Error: 'schema' key must be a dictionary."

    # Check that "strict" is a bool if present
    if "strict" in response_format["json_schema"] and not isinstance(
        response_format["json_schema"]["strict"], bool
    ):
        return "Error: 'strict' key must be a boolean."

    return validate_schema(response_format["json_schema"]["schema"])


def validate_schema(schema, path="root", depth=0, stats=None):
    """
    Validates a JSON schema according to the OpenAI Structured Outputs specification.

    See https://platform.openai.com/docs/guides/structured-outputs for details.

    Note: This code is up to date as of January 21, 2024
    """
    print(f"Validating schema at {path}...")

    # Initialize stats
    if stats is None:
        stats = {
            "total_properties": 0,
            "total_enum_values": 0,
            "total_enum_string_length": 0,
            "total_string_length": 0,
        }

    # Check root object type
    if path == "root" and schema.get("type") != "object":
        return f"Error at {path}: Root schema must be of type 'object'."

    # Check for anyOf at root
    if path == "root" and "anyOf" in schema:
        return f"Error at {path}: Root schema must not use 'anyOf'."

    # Check for required fields
    if schema.get("type") == "object" and "properties" in schema:
        if "required" not in schema or set(schema["required"]) != set(
            schema["properties"].keys()
        ):
            missing_keys = set(schema["properties"].keys()) - set(
                schema.get("required", [])
            )
            return f"Error at {path}: All object properties must be required. Missing keys: {missing_keys}."
        if (
            "additionalProperties" not in schema
            or schema["additionalProperties"] is not False
        ):
            return f"Error at {path}: 'additionalProperties' must be set to false."

    # Check for supported type
    valid_types = {
        "string",
        "number",
        "boolean",
        "integer",
        "object",
        "array",
        "enum",
        "anyOf",
    }
    if "type" in schema:
        schema_type = schema["type"]
        if isinstance(schema_type, list):
            if (
                (len(schema_type) != 2)
                or ("null" not in schema_type)
                or not any(t in valid_types for t in schema_type if t != "null")
            ):
                return f"Error at {path}: Invalid type list {schema_type}. Must contain exactly one valid type and None."
            null_allowed = True
        elif schema_type not in valid_types:
            return f"Error at {path}: Invalid type '{schema_type}'. Must be one of {valid_types}."
        else:
            null_allowed = False

    # Check that enum matches specified type
    if "enum" in schema:
        for enum in schema["enum"]:
            if null_allowed and enum is None:
                continue
            if not null_allowed and enum is None:
                return f"Error at {path}: Enum value cannot be null unless type is [..., null]."

            schema_type = schema.get("type")
            if isinstance(schema_type, list):
                valid_type = next(t for t in schema_type if t != "null")
            else:
                valid_type = schema_type

            if valid_type == "integer" and not isinstance(enum, int):
                return f"Error at {path}: Enum value '{enum}' does not match type 'integer'."
            if valid_type == "number" and not isinstance(enum, (int, float)):
                return f"Error at {path}: Enum value '{enum}' does not match type 'number'."
            if valid_type == "string" and not isinstance(enum, str):
                return f"Error at {path}: Enum value '{enum}' does not match type 'string'."
            if valid_type == "boolean" and not isinstance(enum, bool):
                return f"Error at {path}: Enum value '{enum}' does not match type 'boolean'."
            if valid_type == "object" and not isinstance(enum, dict):
                return f"Error at {path}: Enum value '{enum}' does not match type 'object'."
            if valid_type == "array" and not isinstance(enum, list):
                return (
                    f"Error at {path}: Enum value '{enum}' does not match type 'array'."
                )

    # Check for unsupported keywords based on type
    unsupported_keywords_by_type = {
        "string": ["minLength", "maxLength", "pattern", "format"],
        "number": ["minimum", "maximum", "multipleOf"],
        "integer": ["minimum", "maximum", "multipleOf"],
        "object": [
            "patternProperties",
            "unevaluatedProperties",
            "propertyNames",
            "minProperties",
            "maxProperties",
        ],
        "array": [
            "unevaluatedItems",
            "contains",
            "minContains",
            "maxContains",
            "minItems",
            "maxItems",
            "uniqueItems",
        ],
    }

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_type = next(t for t in schema_type if t != "null")

    if schema_type in unsupported_keywords_by_type:
        for keyword in unsupported_keywords_by_type[schema_type]:
            if keyword in schema:
                return f"Error at {path}: Unsupported keyword '{keyword}' found for type '{schema_type}'."

    # Check for nesting depth
    if depth > 5:
        return f"Error at {path}: Exceeded maximum nesting depth of 5."

    # Check for total properties
    if schema.get("type") == "object":
        stats["total_properties"] += len(schema.get("properties", {}))
        if stats["total_properties"] > 100:
            return "Error: Exceeded maximum of 100 object properties."

    # Check for total string length
    for key in schema.get("properties", {}):
        stats["total_string_length"] += len(key)
    for enum in schema.get("enum", []):
        stats["total_enum_values"] += 1
        stats["total_enum_string_length"] += len(str(enum)) if enum is not None else 4
    if stats["total_string_length"] > 15000:
        return "Error: Exceeded maximum total string length of 15,000 characters."
    if stats["total_enum_values"] > 500:
        return "Error: Exceeded maximum of 500 enum values."
    if stats["total_enum_string_length"] > 7500 and stats["total_enum_values"] > 250:
        return "Error: Exceeded maximum total enum string length of 7,500 characters for more than 250 enum values."

    # Recursively validate nested schemas
    if "properties" in schema:
        for prop, subschema in schema["properties"].items():
            result = validate_schema(
                subschema, path=f"{path}.{prop}", depth=depth + 1, stats=stats
            )
            if result:
                return result

    if "anyOf" in schema:
        for index, subschema in enumerate(schema["anyOf"]):
            result = validate_schema(
                subschema, path=f"{path}.anyOf[{index}]", depth=depth + 1, stats=stats
            )
            if result:
                return result

    if "$defs" in schema:
        for def_name, subschema in schema["$defs"].items():
            result = validate_schema(
                subschema, path=f"{path}.$defs.{def_name}", depth=depth + 1, stats=stats
            )
            if result:
                return result

    if "items" in schema:
        result = validate_schema(
            schema["items"], path=f"{path}.items", depth=depth + 1, stats=stats
        )
        if result:
            return result

    return None


def validate_strict_schema(schema) -> str | None:
    return validate_response_format(
        {"type": "json_schema", "name": "schema", "strict": True, "json_schema": schema}
    )
