# Brushstroke Engine Setup Guide

## Quick Setup

Run the setup script to automatically copy brushstroke_engine code:

```bash
cd third_party/brushstroke_engine
./setup_brushstroke_engine.sh /path/to/brushstroke_engine/brushstroke_engine
```

Or manually copy:

```bash
# From SketchKit root directory
cp -r /path/to/brushstroke_engine/brushstroke_engine/forger third_party/brushstroke_engine/
cp -r /path/to/brushstroke_engine/brushstroke_engine/thirdparty third_party/brushstroke_engine/
```

## Expected Directory Structure

After setup, you should have:

```
third_party/brushstroke_engine/
├── forger/
│   ├── __init__.py
│   ├── ui/
│   │   ├── brush.py          # PaintEngineFactory, PaintingHelper
│   │   └── library.py         # BrushLibrary
│   ├── viz/
│   │   ├── paint_image_main.py
│   │   └── style_transfer.py
│   ├── util/
│   │   └── img_proc.py
│   └── ...
└── thirdparty/
    └── stylegan2_ada_pytorch/
        ├── dnnlib/
        ├── legacy.py
        └── ...
```

## Installation

After copying the code:

```bash
cd /path/to/sketchkit_2026/SketchKit
pip install -e .
```

## Verification

Test that everything works:

```python
import forger
import thirdparty.stylegan2_ada_pytorch.dnnlib as dnnlib
print("✅ Brushstroke engine installed successfully!")
```

## Source

The brushstroke_engine code comes from:
- Repository: https://github.com/NVlabs/brushstroke_engine
- License: See LICENSE file in the original repository

