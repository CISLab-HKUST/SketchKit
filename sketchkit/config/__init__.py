"""
Configuration management for SketchKit.

This module provides unified API key management and configuration utilities.
"""

# High-level API (recommended)
from .manager import api_key, set_api_key, available_providers, is_provider_configured, get_provider_help

# Legacy API (for backward compatibility)
from .api_keys import get_api_key as _get_api_key, set_api_key as _set_api_key, list_available_keys

# Registry and utilities
from .registry import register_provider, list_providers, is_registered

__all__ = [
    # High-level API
    'api_key',
    'set_api_key',
    'available_providers',
    'is_provider_configured',
    'get_provider_help',
    # Registry
    'register_provider',
    'list_providers',
    'is_registered',
    # Legacy API (deprecated, use api_key instead)
    'get_api_key',
    'list_available_keys',
]

# Backward compatibility
def get_api_key(provider: str, required: bool = False, **kwargs):
    """Legacy API - use api_key() instead."""
    import warnings
    warnings.warn(
        "get_api_key() is deprecated. Use api_key() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _get_api_key(provider, required=required, **kwargs)

