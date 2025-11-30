"""
Unified API key management module.

This module provides a secure way to manage API keys for various services.
It supports multiple sources in priority order:
1. Environment variables (highest priority)
2. .env file (project root or user home)
3. Config file (JSON format, optional)
4. Runtime override (via set_api_key)

Security best practices:
- Never commit API keys to version control
- Use environment variables in production
- Use .env file for local development (already in .gitignore)
- Config file is optional and should also be in .gitignore
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List
import warnings

# Try to import python-dotenv for .env file support
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None


# Runtime cache for API keys (set via set_api_key)
_runtime_cache: Dict[str, str] = {}


def _get_project_root() -> Path:
    """Get the project root directory."""
    # Try to find project root by looking for common markers
    current = Path(__file__).resolve()
    
    # Look for .git, setup.py, pyproject.toml, or sketchkit directory
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in ['.git', 'setup.py', 'pyproject.toml', 'sketchkit']):
            return parent
    
    # Fallback to current directory
    return Path.cwd()


def _load_env_file() -> None:
    """Load .env file if available."""
    if not DOTENV_AVAILABLE:
        return
    
    project_root = _get_project_root()
    env_files = [
        project_root / '.env',
        Path.home() / '.sketchkit' / '.env',
    ]
    
    for env_file in env_files:
        if env_file.exists():
            load_dotenv(env_file)
            break


def _get_config_file_path() -> Optional[Path]:
    """Get the path to the config file."""
    project_root = _get_project_root()
    config_files = [
        project_root / '.sketchkit_config.json',
        Path.home() / '.sketchkit' / 'config.json',
    ]
    
    for config_file in config_files:
        if config_file.exists():
            return config_file
    
    return None


def _load_config_file() -> Dict[str, str]:
    """Load API keys from config file."""
    config_file = _get_config_file_path()
    if config_file is None:
        return {}
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            # Extract API keys from config
            api_keys = config.get('api_keys', {})
            if isinstance(api_keys, dict):
                return api_keys
    except (json.JSONDecodeError, IOError) as e:
        warnings.warn(f"Failed to load config file {config_file}: {e}", UserWarning)
    
    return {}


# Load .env file on module import
_load_env_file()


# Standard environment variable names for different services
ENV_VAR_MAPPING = {
    'doubao': 'DOUBAO_API_KEY',
    'doubao-seed3d': 'DOUBAO_API_KEY',
    'openai': 'OPENAI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
    'google': 'GOOGLE_API_KEY',
    'stability': 'STABILITY_API_KEY',
    'replicate': 'REPLICATE_API_KEY',
}


def get_api_key(
    service: str,
    env_var: Optional[str] = None,
    default: Optional[str] = None,
    required: bool = False
) -> Optional[str]:
    """
    Get API key for a service.
    
    Priority order:
    1. Runtime cache (set via set_api_key)
    2. Environment variable
    3. .env file (via environment variable)
    4. Config file
    5. Default value (if provided)
    
    Args:
        service: Service name (e.g., 'doubao', 'openai')
        env_var: Custom environment variable name (if None, uses standard mapping)
        default: Default value if key not found (not recommended for production)
        required: If True, raises ValueError when key is not found
    
    Returns:
        API key string, or None if not found and not required
    
    Raises:
        ValueError: If required=True and key is not found
    
    Examples:
        >>> # Get Doubao API key
        >>> key = get_api_key('doubao')
        >>> 
        >>> # Get with custom env var
        >>> key = get_api_key('custom-service', env_var='CUSTOM_API_KEY')
        >>> 
        >>> # Require key (raises error if not found)
        >>> key = get_api_key('doubao', required=True)
    """
    # 1. Check runtime cache
    if service in _runtime_cache:
        return _runtime_cache[service]
    
    # 2. Check environment variable
    if env_var is None:
        env_var = ENV_VAR_MAPPING.get(service.lower(), f'{service.upper()}_API_KEY')
    
    api_key = os.environ.get(env_var)
    if api_key:
        return api_key.strip()
    
    # 3. Config file (already loaded .env via load_dotenv, so env vars are checked)
    config_keys = _load_config_file()
    if service in config_keys:
        api_key = config_keys[service]
        if api_key:
            return api_key.strip()
    
    # 4. Default value
    if default is not None:
        if required:
            warnings.warn(
                f"Using default API key for {service}. This is not recommended for production.",
                UserWarning
            )
        return default
    
    # 5. Not found
    if required:
        sources = [
            f"environment variable '{env_var}'",
            ".env file",
            "config file",
        ]
        raise ValueError(
            f"API key for service '{service}' not found. "
            f"Please set it via one of: {', '.join(sources)}, "
            f"or use set_api_key('{service}', 'your_key')"
        )
    
    return None


def set_api_key(service: str, api_key: str) -> None:
    """
    Set API key in runtime cache (takes highest priority).
    
    This is useful for programmatic configuration or testing.
    The key will be used for the current Python session only.
    
    Args:
        service: Service name (e.g., 'doubao', 'openai')
        api_key: API key value
    
    Examples:
        >>> set_api_key('doubao', 'your-api-key-here')
        >>> key = get_api_key('doubao')  # Returns 'your-api-key-here'
    """
    _runtime_cache[service.lower()] = api_key.strip()


def list_available_keys() -> List[str]:
    """
    List all services that have API keys available.
    
    Returns:
        List of service names that have keys configured
    
    Examples:
        >>> services = list_available_keys()
        >>> print(f"Configured services: {services}")
    """
    available = []
    
    # Check environment variables
    for service, env_var in ENV_VAR_MAPPING.items():
        if os.environ.get(env_var):
            available.append(service)
    
    # Check config file
    config_keys = _load_config_file()
    for service in config_keys:
        if service not in available and config_keys[service]:
            available.append(service)
    
    # Check runtime cache
    for service in _runtime_cache:
        if service not in available:
            available.append(service)
    
    return sorted(set(available))

