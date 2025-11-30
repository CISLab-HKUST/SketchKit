"""
Configuration loading logic.

This module implements the loading logic with priority:
1. Environment variables (highest)
2. .env file
3. ~/.sketchkit/config.yml
4. Project .sketchkit_config.yml
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

# Try to import python-dotenv
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None

# Try to import YAML
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

from .paths import get_config_path, get_project_config_path, get_env_file_path
from .registry import get_provider_info, API_KEY_REGISTRY

# Runtime cache for API keys (set via manager.set_api_key)
_runtime_cache: Dict[str, str] = {}


def _load_env_file() -> None:
    """Load .env file if available."""
    if not DOTENV_AVAILABLE:
        return
    
    env_path = get_env_file_path()
    if env_path:
        load_dotenv(env_path, override=False)


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        Configuration dictionary
    
    Raises:
        ValueError: If YAML is not available or file is invalid
    """
    if not YAML_AVAILABLE:
        return {}
    
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            if config is None:
                return {}
            return config
    except (yaml.YAMLError, IOError) as e:
        warnings.warn(f"Failed to load config file {config_path}: {e}", UserWarning)
        return {}


def load_from_env(provider: str) -> Optional[str]:
    """
    Load API key from environment variables.
    
    Args:
        provider: Provider name
    
    Returns:
        API key if found, None otherwise
    
    Examples:
        >>> os.environ['OPENAI_API_KEY'] = 'sk-xxx'
        >>> key = load_from_env('openai')
        >>> print(key)
        sk-xxx
    """
    provider_info = get_provider_info(provider)
    if provider_info is None:
        return None
    
    for env_var in provider_info['env']:
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key.strip()
    
    return None


def load_from_config_file(provider: str) -> Optional[str]:
    """
    Load API key from config file.
    
    Checks both global and project config files.
    
    Args:
        provider: Provider name
    
    Returns:
        API key if found, None otherwise
    
    Examples:
        >>> key = load_from_config_file('openai')
    """
    provider_info = get_provider_info(provider)
    if provider_info is None:
        return None
    
    config_key = provider_info['config_key']
    
    # Try project config first
    project_config_path = get_project_config_path()
    if project_config_path:
        config = _load_yaml_config(project_config_path)
        if config_key in config.get('api_keys', {}):
            api_key = config['api_keys'][config_key]
            if api_key:
                return str(api_key).strip()
    
    # Try global config
    global_config_path = get_config_path()
    if global_config_path.exists():
        config = _load_yaml_config(global_config_path)
        if config_key in config.get('api_keys', {}):
            api_key = config['api_keys'][config_key]
            if api_key:
                return str(api_key).strip()
    
    return None


def get_api_key(
    provider: str,
    required: bool = False,
    default: Optional[str] = None
) -> Optional[str]:
    """
    Get API key for a provider (core entry point).
    
    Priority order:
    1. Runtime cache (set via manager.set_api_key)
    2. Environment variables
    3. .env file (via environment variables)
    4. Config file (~/.sketchkit/config.yml or project config)
    5. Default value (if provided)
    
    Args:
        provider: Provider name
        required: If True, raises ValueError when key is not found
        default: Default value if key not found (not recommended)
    
    Returns:
        API key string, or None if not found and not required
    
    Raises:
        ValueError: If required=True and key is not found
    
    Examples:
        >>> key = get_api_key('openai')
        >>> key = get_api_key('openai', required=True)
    """
    # 1. Check runtime cache
    if provider.lower() in _runtime_cache:
        return _runtime_cache[provider.lower()]
    
    # 2. Load .env file (if not already loaded)
    _load_env_file()
    
    # 3. Check environment variables
    api_key = load_from_env(provider)
    if api_key:
        return api_key
    
    # 4. Check config file
    api_key = load_from_config_file(provider)
    if api_key:
        return api_key
    
    # 5. Default value
    if default is not None:
        if required:
            warnings.warn(
                f"Using default API key for {provider}. "
                "This is not recommended for production.",
                UserWarning
            )
        return default
    
    # 6. Not found
    if required:
        provider_info = get_provider_info(provider)
        if provider_info is None:
            raise ValueError(
                f"Provider '{provider}' is not registered. "
                f"Available providers: {', '.join(sorted(API_KEY_REGISTRY.keys()))}"
            )
        
        env_vars = ', '.join(provider_info['env'])
        raise ValueError(
            f"API key for provider '{provider}' not found.\n"
            f"Please set it via one of:\n"
            f"  1. Environment variable: {env_vars}\n"
            f"  2. .env file: {env_vars}=your_key\n"
            f"  3. Config file: ~/.sketchkit/config.yml\n"
            f"  4. CLI: sketchkit configure {provider}\n"
            f"  5. Runtime: sketchkit.config.set_api_key('{provider}', 'your_key')"
        )
    
    return None


def set_runtime_key(provider: str, api_key: str) -> None:
    """
    Set API key in runtime cache (internal use).
    
    Args:
        provider: Provider name
        api_key: API key value
    """
    _runtime_cache[provider.lower()] = api_key.strip()

