# Brushstroke Engine Integration

This module provides integration with the Neural Brushstroke Engine (NeuBE) for stylizing sketches.

## Features

- Automatic model downloading
- Support for style1 and style2 models
- Multiple brushstroke styles
- Simple API similar to `sketch2image` and `sketch2model`

## Usage

### Basic Usage

```python
from sketchkit.brushstrokeengine import BrushstrokeStylizer
from sketchkit.datasets import PhotoSketching

# Initialize stylizer (will auto-download models if needed)
stylizer = BrushstrokeStylizer(
    model_name="style2",  # or "style1"
    style_id="playdoh10",  # style ID
    auto_download=True,  # automatically download models if not found
)

# Load a sketch
dataset = PhotoSketching()
sketch = dataset[0]

# Stylize the sketch
stylized_image = stylizer.run(sketch, canvas_size=512)

# Save the result
stylized_image.save("output.png")
```

### Manual Model Download

```python
from sketchkit.brushstrokeengine.models import download_models

# Download models to default location
download_models()

# Or specify a custom location
download_models(models_dir="/path/to/models")
```

### Available Styles

Common style IDs include:
- `playdoh10`
- `misc05`
- `candy05`
- And many more...

To see available styles, check the brush library files in `models/neube/{model_name}/brush_libs/`.

## Requirements

### Python Dependencies

All required Python dependencies are included in `pyproject.toml`:
- `torch>=2.8.0`
- `scikit-image>=0.25.2`
- `gdown>=5.2.0`
- And other standard dependencies

**Note**: `tqdm>=4.56.0` is required by the brushstroke_engine package. 
If you encounter an import error, install it with: `pip install tqdm>=4.56.0`

### Brushstroke Engine Code (Required)

**Why we need the brushstroke_engine code:**

The brushstroke stylization requires complex modules that are impractical to reimplement:
- **StyleGAN2 implementation** (`thirdparty.stylegan2_ada_pytorch`) - GAN model loading and inference
- **PaintEngineFactory** - Creates the GAN engine from checkpoints
- **PaintingHelper** - Renders brushstrokes using the GAN
- **BrushLibrary** - Loads and manages style libraries
- **Many other internal modules** with deep dependencies

**Setup:**

The brushstroke_engine code should be placed in `third_party/brushstroke_engine/`:

```bash
# Copy from brushstroke_engine repository
cp -r /path/to/brushstroke_engine/brushstroke_engine/forger third_party/brushstroke_engine/
cp -r /path/to/brushstroke_engine/brushstroke_engine/thirdparty third_party/brushstroke_engine/
```

After copying, install sketchkit:

```bash
pip install -e .
```

The `forger` and `thirdparty` modules will be automatically included via `setup.py` configuration.

## Model Structure

After downloading, models are stored in:
```
sketchkit/brushstrokeengine/models/
└── neube/
    ├── style1/
    │   ├── snapshot.pkl
    │   └── brush_libs/
    └── style2/
        ├── snapshot.pkl
        └── brush_libs/
```

## API Reference

### BrushstrokeStylizer

#### `__init__(model_name="style2", style_id="playdoh10", models_dir=None, device="cuda", auto_download=True)`

Initialize the stylizer.

- `model_name`: Model to use ('style1' or 'style2')
- `style_id`: Style ID for stylization
- `models_dir`: Custom models directory (optional)
- `device`: Device to use (for future use)
- `auto_download`: Automatically download models if not found

#### `run(sketch, output_path=None, canvas_size=512, style_id=None, library_path=None, **kwargs)`

Stylize a sketch.

- `sketch`: Input Sketch object
- `output_path`: Output file path (optional)
- `canvas_size`: Canvas size for rendering
- `style_id`: Style ID (overrides instance default)
- `library_path`: Custom brush library path (optional)
- `**kwargs`: Additional parameters:
  - `feature_blending_level` (int): Default 2
  - `color_mode` (str): Default '1'
  - `crop_margin` (int): Default 10
  - `on_white` (bool): Default True
  - `render_mode` (str): Optional

Returns: PIL Image of the stylized result

