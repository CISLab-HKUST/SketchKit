"""
API Provider registry.

This module maintains a registry of API providers and their configuration
details, including environment variable names and config file keys.
"""

from typing import Dict, List, Optional, Any


# API Provider Registry
# Each entry contains:
# - env: List of environment variable names to check
# - config_key: Key name in config file
# - description: Human-readable description
API_KEY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "openai": {
        "env": ["OPENAI_API_KEY"],
        "config_key": "openai",
        "description": "OpenAI API key for GPT models",
    },
    "anthropic": {
        "env": ["ANTHROPIC_API_KEY"],
        "config_key": "anthropic",
        "description": "Anthropic API key for Claude models",
    },
    "google": {
        "env": ["GOOGLE_API_KEY"],
        "config_key": "google",
        "description": "Google API key",
    },
    "huggingface": {
        "env": ["HF_TOKEN", "HUGGINGFACE_API_KEY"],
        "config_key": "huggingface",
        "description": "Hugging Face API token",
    },
    "stability": {
        "env": ["STABILITY_API_KEY", "STABILITYAI_API_KEY"],
        "config_key": "stability",
        "description": "Stability AI API key",
    },
    "replicate": {
        "env": ["REPLICATE_API_TOKEN", "REPLICATE_API_KEY"],
        "config_key": "replicate",
        "description": "Replicate API token",
    },
    "doubao": {
        "env": ["DOUBAO_API_KEY"],
        "config_key": "doubao",
        "description": "Doubao (ByteDance) API key",
    },
    "doubao-seed3d": {
        "env": ["DOUBAO_API_KEY"],
        "config_key": "doubao",
        "description": "Doubao Seed3D API key (alias for doubao)",
    },
}


def register_provider(
    name: str,
    env_vars: List[str],
    config_key: Optional[str] = None,
    description: Optional[str] = None
) -> None:
    """
    Register a new API provider.
    
    Args:
        name: Provider name (e.g., 'custom-service')
        env_vars: List of environment variable names to check
        config_key: Key name in config file (defaults to name if None)
        description: Human-readable description
    
    Examples:
        >>> register_provider(
        ...     'custom-service',
        ...     env_vars=['CUSTOM_API_KEY'],
        ...     description='Custom service API key'
        ... )
    """
    if config_key is None:
        config_key = name
    
    API_KEY_REGISTRY[name.lower()] = {
        "env": env_vars,
        "config_key": config_key,
        "description": description or f"{name} API key",
    }


def get_provider_info(provider: str) -> Optional[Dict[str, Any]]:
    """
    Get provider information from registry.
    
    Args:
        provider: Provider name
    
    Returns:
        Provider info dict, or None if not found
    
    Examples:
        >>> info = get_provider_info('openai')
        >>> print(info['env'])
        ['OPENAI_API_KEY']
    """
    return API_KEY_REGISTRY.get(provider.lower())


def list_providers() -> List[str]:
    """
    List all registered providers.
    
    Returns:
        List of provider names
    
    Examples:
        >>> providers = list_providers()
        >>> print(providers)
        ['openai', 'anthropic', 'google', ...]
    """
    return sorted(API_KEY_REGISTRY.keys())


def is_registered(provider: str) -> bool:
    """
    Check if a provider is registered.
    
    Args:
        provider: Provider name
    
    Returns:
        True if registered, False otherwise
    
    Examples:
        >>> is_registered('openai')
        True
        >>> is_registered('unknown')
        False
    """
    return provider.lower() in API_KEY_REGISTRY

