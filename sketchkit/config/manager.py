"""
High-level API key management interface.

This module provides a simple, high-level interface for managing API keys.
"""

from typing import Optional, List
from .loader import get_api_key as _get_api_key, set_runtime_key
from .registry import list_providers, is_registered, get_provider_info


def api_key(provider: str, required: bool = False) -> Optional[str]:
    """
    Get API key for a provider (high-level interface).
    
    This is the recommended way to get API keys in your code.
    
    Args:
        provider: Provider name (e.g., 'openai', 'doubao')
        required: If True, raises ValueError when key is not found
    
    Returns:
        API key string, or None if not found and not required
    
    Raises:
        ValueError: If required=True and key is not found
    
    Examples:
        >>> from sketchkit.config import api_key
        >>> 
        >>> # Get OpenAI API key
        >>> key = api_key('openai', required=True)
        >>> 
        >>> # Get Doubao API key (returns None if not found)
        >>> key = api_key('doubao')
    """
    return _get_api_key(provider, required=required)


def set_api_key(provider: str, api_key: str) -> None:
    """
    Set API key in runtime cache.
    
    This is useful for programmatic configuration or testing.
    The key will be used for the current Python session only.
    
    Args:
        provider: Provider name
        api_key: API key value
    
    Examples:
        >>> from sketchkit.config import set_api_key
        >>> 
        >>> set_api_key('openai', 'sk-xxx')
        >>> key = api_key('openai')  # Returns 'sk-xxx'
    """
    set_runtime_key(provider, api_key)


def available_providers() -> List[str]:
    """
    Get list of all registered providers.
    
    Returns:
        List of provider names
    
    Examples:
        >>> from sketchkit.config import available_providers
        >>> 
        >>> providers = available_providers()
        >>> print(providers)
        ['anthropic', 'doubao', 'google', ...]
    """
    return list_providers()


def is_provider_configured(provider: str) -> bool:
    """
    Check if a provider has an API key configured.
    
    Args:
        provider: Provider name
    
    Returns:
        True if key is available, False otherwise
    
    Examples:
        >>> from sketchkit.config import is_provider_configured
        >>> 
        >>> if is_provider_configured('openai'):
        ...     print("OpenAI is configured")
    """
    if not is_registered(provider):
        return False
    
    key = api_key(provider, required=False)
    return key is not None


def get_provider_help(provider: str) -> str:
    """
    Get help text for configuring a provider.
    
    Args:
        provider: Provider name
    
    Returns:
        Help text string
    
    Examples:
        >>> from sketchkit.config import get_provider_help
        >>> 
        >>> help_text = get_provider_help('openai')
        >>> print(help_text)
    """
    provider_info = get_provider_info(provider)
    if provider_info is None:
        return f"Provider '{provider}' is not registered."
    
    env_vars = ', '.join(provider_info['env'])
    config_key = provider_info['config_key']
    
    return (
        f"To configure {provider}:\n"
        f"  1. Environment variable: {env_vars}\n"
        f"  2. .env file: {env_vars}=your_key\n"
        f"  3. Config file: ~/.sketchkit/config.yml (key: {config_key})\n"
        f"  4. CLI: sketchkit configure {provider}\n"
        f"  5. Runtime: sketchkit.config.set_api_key('{provider}', 'your_key')"
    )

