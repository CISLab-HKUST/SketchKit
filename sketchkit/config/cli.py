"""
CLI commands for API key configuration.

This module provides the `sketchkit configure` command for managing API keys.
"""

import click
import getpass
from pathlib import Path
from typing import Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

from .paths import get_config_path, get_config_dir
from .registry import get_provider_info, list_providers, is_registered
from .manager import api_key, get_provider_help


@click.group()
def cli():
    """SketchKit configuration management."""
    pass


@cli.command()
@click.argument('provider')
def configure(provider: str):
    """
    Configure API key for a provider.
    
    This command will prompt you to enter the API key (hidden input)
    and save it to ~/.sketchkit/config.yml.
    
    Examples:
        sketchkit configure openai
        sketchkit configure doubao
    """
    if not YAML_AVAILABLE:
        click.echo("Error: PyYAML is required for configuration. Install it with: pip install pyyaml", err=True)
        return
    
    # Check if provider is registered
    if not is_registered(provider):
        click.echo(f"Error: Provider '{provider}' is not registered.", err=True)
        click.echo(f"\nAvailable providers: {', '.join(list_providers())}")
        return
    
    provider_info = get_provider_info(provider)
    if provider_info is None:
        click.echo(f"Error: Could not get provider info for '{provider}'", err=True)
        return
    
    # Check if already configured
    existing_key = api_key(provider, required=False)
    if existing_key:
        click.echo(f"\n{provider} is already configured.")
        if click.confirm("Do you want to update it?"):
            pass
        else:
            click.echo("Cancelled.")
            return
    
    # Prompt for API key
    click.echo(f"\nConfiguring {provider} API key")
    click.echo(f"Description: {provider_info.get('description', 'N/A')}")
    click.echo(f"Environment variable: {', '.join(provider_info['env'])}")
    click.echo()
    
    api_key_value = getpass.getpass("Enter API key (input will be hidden): ")
    
    if not api_key_value.strip():
        click.echo("Error: API key cannot be empty.", err=True)
        return
    
    # Save to config file
    config_path = get_config_path()
    config_dir = get_config_dir()
    
    # Load existing config
    config = {}
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            click.echo(f"Warning: Could not load existing config: {e}", err=True)
    
    # Update config
    if 'api_keys' not in config:
        config['api_keys'] = {}
    
    config_key = provider_info['config_key']
    config['api_keys'][config_key] = api_key_value.strip()
    
    # Save config
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Set appropriate permissions (read/write for user only)
        import os
        if os.name != 'nt':  # Unix-like systems
            os.chmod(config_path, 0o600)  # rw-------
        
        click.echo(f"\n✓ Successfully configured {provider} API key")
        click.echo(f"  Saved to: {config_path}")
    except Exception as e:
        click.echo(f"Error: Failed to save config: {e}", err=True)
        return


@cli.command('list')
def list_providers_cmd():
    """List all registered providers and their configuration status."""
    providers = list_providers()
    
    if not providers:
        click.echo("No providers registered.")
        return
    
    click.echo("Registered providers:\n")
    
    for provider in sorted(providers):
        provider_info = get_provider_info(provider)
        if provider_info is None:
            continue
        
        configured = api_key(provider, required=False) is not None
        status = "✓ Configured" if configured else "✗ Not configured"
        
        click.echo(f"  {provider:20} {status}")
        click.echo(f"    {provider_info.get('description', 'N/A')}")
        click.echo(f"    Env vars: {', '.join(provider_info['env'])}")
        click.echo()


@cli.command('show')
@click.argument('provider')
def show_provider(provider: str):
    """Show configuration help for a provider."""
    if not is_registered(provider):
        click.echo(f"Error: Provider '{provider}' is not registered.", err=True)
        click.echo(f"\nAvailable providers: {', '.join(list_providers())}")
        return
    
    help_text = get_provider_help(provider)
    click.echo(help_text)
    
    # Show current status
    key = api_key(provider, required=False)
    if key:
        masked_key = key[:8] + '...' + key[-4:] if len(key) > 12 else '***'
        click.echo(f"\nCurrent status: Configured (key: {masked_key})")
    else:
        click.echo("\nCurrent status: Not configured")


@cli.command('test')
@click.argument('provider')
def test_provider(provider: str):
    """Test if a provider's API key is configured and accessible."""
    if not is_registered(provider):
        click.echo(f"Error: Provider '{provider}' is not registered.", err=True)
        return
    
    key = api_key(provider, required=False)
    if key is None:
        click.echo(f"✗ {provider} is not configured.")
        click.echo(f"\n{get_provider_help(provider)}")
        return
    
    masked_key = key[:8] + '...' + key[-4:] if len(key) > 12 else '***'
    click.echo(f"✓ {provider} is configured (key: {masked_key})")
    click.echo("Note: This only checks if the key is available, not if it's valid.")


if __name__ == '__main__':
    cli()

