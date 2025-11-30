"""
Configuration file path management.

This module provides utilities for managing configuration file paths,
including automatic directory creation and path resolution.
"""

import os
from pathlib import Path
from typing import Optional


def get_config_dir() -> Path:
    """
    Get the global configuration directory.
    
    Returns:
        Path to ~/.sketchkit/ directory
    
    Examples:
        >>> config_dir = get_config_dir()
        >>> print(config_dir)
        /home/user/.sketchkit
    """
    home = Path.home()
    config_dir = home / '.sketchkit'
    
    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Set appropriate permissions (read/write for user only)
    if os.name != 'nt':  # Unix-like systems
        os.chmod(config_dir, 0o700)  # rwx------
    
    return config_dir


def get_config_path() -> Path:
    """
    Get the path to the global configuration file.
    
    Returns:
        Path to ~/.sketchkit/config.yml
    
    Examples:
        >>> config_path = get_config_path()
        >>> print(config_path)
        /home/user/.sketchkit/config.yml
    """
    config_dir = get_config_dir()
    return config_dir / 'config.yml'


def get_project_config_path() -> Optional[Path]:
    """
    Get the path to project-local configuration file.
    
    Looks for .sketchkit_config.yml in project root.
    
    Returns:
        Path to project config file, or None if not found
    
    Examples:
        >>> project_config = get_project_config_path()
        >>> if project_config:
        ...     print(f"Found: {project_config}")
    """
    # Try to find project root
    current = Path.cwd()
    
    # Look for common project markers
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in ['.git', 'setup.py', 'pyproject.toml']):
            config_file = parent / '.sketchkit_config.yml'
            if config_file.exists():
                return config_file
    
    return None


def get_env_file_path() -> Optional[Path]:
    """
    Get the path to .env file.
    
    Checks project root first, then home directory.
    
    Returns:
        Path to .env file, or None if not found
    
    Examples:
        >>> env_path = get_env_file_path()
        >>> if env_path:
        ...     print(f"Found .env at: {env_path}")
    """
    # Check project root
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in ['.git', 'setup.py', 'pyproject.toml']):
            env_file = parent / '.env'
            if env_file.exists():
                return env_file
    
    # Check home directory
    home_env = Path.home() / '.sketchkit' / '.env'
    if home_env.exists():
        return home_env
    
    return None

