# Sketch2Model Module

This module provides a unified API interface and implementations for three sketch-to-3D model conversion methods.

## Architecture

The `sketchkit/sketch2model/` module is a **self-contained module** that includes both the API interface and all implementation code:

- **API Layer**: `sketchkit/sketch2model/__init__.py` - Unified interface
- **Implementations**: 
  - `sketchkit/sketch2model/teddy_monstermash.py` - Teddy/MonsterMash method
  - `sketchkit/sketch2model/sens_minimal.py` - SENS method
  - `sketchkit/sketch2model/doubao.py` - Doubao-Seed3D API method

## Module Structure

```
sketchkit/sketch2model/
├── __init__.py              # Unified API interface
├── teddy_monstermash.py     # Teddy/MonsterMash implementation
├── sens_minimal.py          # SENS implementation
├── doubao.py                # Doubao-Seed3D implementation
└── [documentation files]    # PR docs, usage guides, etc.
```

## Why This Design?

1. **Self-Contained**: All code is in one place, making it easier to maintain and distribute
2. **Clean API**: Users get a simple `from sketchkit.sketch2model import sketch_to_3d` interface
3. **Modular**: Each method is in its own file, making it easy to understand and modify
4. **Standard Structure**: Follows standard Python package structure conventions

## Usage

```python
from sketchkit.sketch2model import sketch_to_3d

# Use unified interface
output = sketch_to_3d(sketch, method='teddy', output_path='output.obj')
output = sketch_to_3d(sketch, method='sens', output_path='output.obj')
output = sketch_to_3d(sketch, method='doubao-seed3d', output_path='output.glb', api_key='your_key')
```

## Files in This Module

- `__init__.py` - Unified API interface (imports from `implemented_methods`)
- Documentation files (PR_DESCRIPTION_EN.md, etc.)

## Implementation Files (in `implemented_methods/`)

These files are **not** part of this PR but are required dependencies:
- `teddy_monstermash.py` - Teddy implementation
- `sens_minimal.py` - SENS implementation  
- `sketch2model.py` - Full converter with Doubao method
- `external/` - External dependencies (MonsterMash C++ code, SENS source code)

