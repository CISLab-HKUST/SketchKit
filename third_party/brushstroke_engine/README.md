# Brushstroke Engine Integration

This directory should contain the brushstroke_engine codebase for stylization functionality.

## Setup Instructions

To set up brushstroke_engine:

1. **Copy the brushstroke_engine codebase** to this directory:

```bash
# From the brushstroke_engine repository root
cp -r /path/to/brushstroke_engine/brushstroke_engine/forger third_party/brushstroke_engine/
cp -r /path/to/brushstroke_engine/brushstroke_engine/thirdparty third_party/brushstroke_engine/
```

Or if you have the brushstroke_engine repository cloned:

```bash
cd /path/to/brushstroke_engine/brushstroke_engine
cp -r forger /hpc2hdd/home/xliao819/developer/sketchkit_2026/SketchKit/third_party/brushstroke_engine/
cp -r thirdparty /hpc2hdd/home/xliao819/developer/sketchkit_engine/brushstroke_engine/
```

2. **Verify the structure**:

```
third_party/brushstroke_engine/
├── forger/
│   ├── __init__.py
│   ├── ui/
│   │   ├── brush.py          # PaintEngineFactory, PaintingHelper
│   │   └── library.py        # BrushLibrary
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

3. **Install sketchkit**:

```bash
cd /hpc2hdd/home/xliao819/developer/sketchkit_2026/SketchKit
pip install -e .
```

4. **Verify installation**:

```python
import forger
import thirdparty.stylegan2_ada_pytorch.dnnlib as dnnlib
print("✅ Brushstroke engine installed successfully!")
```

## What's Included

- **forger/**: The core brushstroke engine module
  - `forger.ui.brush`: PaintEngineFactory, PaintingHelper, GanBrushOptions
  - `forger.ui.library`: BrushLibrary for style management
  - `forger.viz.style_transfer`: Patch generation for large images
  - `forger.util.img_proc`: Image processing utilities

- **thirdparty/stylegan2_ada_pytorch/**: StyleGAN2 implementation
  - Required for loading and running GAN checkpoints
  - Used by PaintEngineFactory to load models

## Note

The brushstroke_engine code is included as-is from the original repository.
No modifications are made to the core forger code - it's used directly.

