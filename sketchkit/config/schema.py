"""
Configuration file schema definitions.

This module defines the structure and validation for configuration files.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path


# Configuration schema structure
CONFIG_SCHEMA = {
    "api_keys": {
        "type": "dict",
        "description": "API keys for various providers",
        "keys": {
            "openai": {
                "type": "string",
                "description": "OpenAI API key",
                "pattern": "^sk-",
            },
            "anthropic": {
                "type": "string",
                "description": "Anthropic API key",
            },
            "google": {
                "type": "string",
                "description": "Google API key",
            },
            "huggingface": {
                "type": "string",
                "description": "Hugging Face API token",
                "pattern": "^hf_",
            },
            "stability": {
                "type": "string",
                "description": "Stability AI API key",
            },
            "replicate": {
                "type": "string",
                "description": "Replicate API token",
            },
            "doubao": {
                "type": "string",
                "description": "Doubao API key",
            },
        },
    },
}


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration structure.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of validation errors (empty if valid)
    
    Examples:
        >>> config = {"api_keys": {"openai": "sk-xxx"}}
        >>> errors = validate_config(config)
        >>> if errors:
        ...     print("Validation errors:", errors)
    """
    errors = []
    
    if not isinstance(config, dict):
        errors.append("Config must be a dictionary")
        return errors
    
    # Check api_keys section
    if "api_keys" in config:
        api_keys = config["api_keys"]
        if not isinstance(api_keys, dict):
            errors.append("api_keys must be a dictionary")
        else:
            # Validate individual keys
            for key, value in api_keys.items():
                if not isinstance(value, str):
                    errors.append(f"api_keys.{key} must be a string")
                elif not value.strip():
                    errors.append(f"api_keys.{key} cannot be empty")
    
    return errors


def get_config_template() -> Dict[str, Any]:
    """
    Get a template configuration dictionary.
    
    Returns:
        Template configuration
    
    Examples:
        >>> template = get_config_template()
        >>> print(template)
    """
    return {
        "api_keys": {
            "openai": "YOUR_OPENAI_KEY",
            "anthropic": "YOUR_ANTHROPIC_KEY",
            "google": "YOUR_GOOGLE_KEY",
            "huggingface": "YOUR_HF_TOKEN",
            "stability": "YOUR_STABILITY_API_KEY",
            "replicate": "YOUR_REPLICATE_API_TOKEN",
            "doubao": "YOUR_DOUBAO_API_KEY",
        },
    }

