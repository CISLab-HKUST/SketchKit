# API Key Management

This module provides a unified and secure way to manage API keys for various services used in SketchKit.

## Features

- **Multiple source support**: Environment variables, `.env` files, config files, and runtime configuration
- **Priority-based loading**: Clear priority order ensures predictable behavior
- **Security best practices**: Never commit keys to version control
- **Easy to use**: Simple API for getting and setting keys
- **Extensible**: Easy to add new services

## Quick Start

### Method 1: Environment Variable (Recommended for Production)

```bash
# Linux/macOS
export DOUBAO_API_KEY='your-api-key-here'

# Windows (PowerShell)
$env:DOUBAO_API_KEY='your-api-key-here'

# Windows (CMD)
set DOUBAO_API_KEY=your-api-key-here
```

### Method 2: .env File (Recommended for Local Development)

Create a `.env` file in the project root:

```bash
# .env
DOUBAO_API_KEY=your-api-key-here
OPENAI_API_KEY=your-openai-key
```

**Note**: `.env` files are already in `.gitignore` and will not be committed.

### Method 3: Config File (Optional)

Create `.sketchkit_config.json` in the project root:

```json
{
  "api_keys": {
    "doubao": "your-doubao-key",
    "openai": "your-openai-key"
  }
}
```

**Note**: Config files should also be in `.gitignore`.

### Method 4: Runtime Configuration (For Testing)

```python
from sketchkit.config import set_api_key

set_api_key('doubao', 'your-api-key-here')
```

## Usage

### Getting API Keys

```python
from sketchkit.config import get_api_key

# Get Doubao API key
key = get_api_key('doubao')

# Get with custom environment variable
key = get_api_key('custom-service', env_var='CUSTOM_API_KEY')

# Require key (raises error if not found)
key = get_api_key('doubao', required=True)

# With default (not recommended for production)
key = get_api_key('doubao', default='fallback-key')
```

### Setting API Keys Programmatically

```python
from sketchkit.config import set_api_key

set_api_key('doubao', 'your-api-key-here')
```

### Listing Available Keys

```python
from sketchkit.config import list_available_keys

services = list_available_keys()
print(f"Configured services: {services}")
```

## Priority Order

API keys are loaded in the following priority order (highest to lowest):

1. **Runtime cache** (set via `set_api_key()`)
2. **Environment variables**
3. **.env file** (loaded via `python-dotenv`)
4. **Config file** (`.sketchkit_config.json`)
5. **Default value** (if provided, with warning)

## Supported Services

The module includes standard mappings for common services:

- `doubao` / `doubao-seed3d` → `DOUBAO_API_KEY`
- `openai` → `OPENAI_API_KEY`
- `anthropic` → `ANTHROPIC_API_KEY`
- `google` → `GOOGLE_API_KEY`
- `stability` → `STABILITY_API_KEY`
- `replicate` → `REPLICATE_API_KEY`

You can use any service name - if no standard mapping exists, it will look for `{SERVICE}_API_KEY` environment variable.

## Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** in production environments
3. **Use .env files** for local development (already in `.gitignore`)
4. **Rotate keys regularly** if compromised
5. **Use different keys** for different environments (dev, staging, prod)
6. **Restrict key permissions** when possible (IP whitelisting, rate limiting)

## Dependencies

The module optionally uses `python-dotenv` for `.env` file support:

```bash
pip install python-dotenv
```

If `python-dotenv` is not installed, the module will still work but won't load `.env` files automatically.

## Examples

### Example 1: Using in Your Code

```python
from sketchkit.config import get_api_key
from sketchkit.sketch2model import Modeler

# Get API key
api_key = get_api_key('doubao', required=True)

# Use with Modeler
modeler = Modeler(method='doubao-seed3d')
output = modeler.run(sketch, api_key=api_key, output_path='output.glb')
```

### Example 2: Multiple Services

```python
from sketchkit.config import get_api_key

doubao_key = get_api_key('doubao', required=True)
openai_key = get_api_key('openai', required=True)

# Use keys...
```

### Example 3: Custom Service

```python
from sketchkit.config import get_api_key

# Custom service with custom env var
custom_key = get_api_key('my-service', env_var='MY_SERVICE_API_KEY', required=True)
```

## Troubleshooting

### Key Not Found Error

If you get a "key not found" error:

1. Check that the environment variable is set: `echo $DOUBAO_API_KEY`
2. Check that `.env` file exists and has the correct format
3. Check that config file exists and has correct JSON format
4. Use `list_available_keys()` to see what's configured

### .env File Not Loading

If `.env` file is not loading:

1. Install `python-dotenv`: `pip install python-dotenv`
2. Check file location (should be in project root)
3. Check file format (key=value, no quotes needed)

### Config File Not Loading

If config file is not loading:

1. Check file location (`.sketchkit_config.json` in project root)
2. Check JSON format is valid
3. Check that `api_keys` key exists in JSON

