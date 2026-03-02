# Stylization

The stylization module provides functionality to convert vector sketches into stylized raster images using various brushstroke styles. This enables artistic rendering of sketches with different visual styles, textures, and brush effects.

**Source**: `sketchkit/stylization/`

## Introduction of the Task

Sketch stylization is the process of converting clean line drawings into stylized images that mimic various artistic media and brushstroke styles. The task involves learning a latent space of interactive drawing tools from unlabeled images of drawing media, enabling automatic stylization of sketches with diverse visual styles. The stylization process preserves the geometric structure of the input sketch while applying learned brushstroke textures, colors, and artistic effects.

## Available Methods

The module currently supports the following method:

- **Stylizer**: Unified interface for stylizing sketches using various brushstroke styles.

## Details of Each Method

### BrushstrokesEngine

The method implements [Neural Brushstroke Engine: Learning a Latent Space of Interactive Drawing Tools](https://research.nvidia.com/labs/toronto-ai/brushstroke_engine/) from *SIGGRAPH Asia 2022*. It is a GAN-based system that learns to mimic drawing media styles from unlabeled images. The model extends StyleGAN2 architecture to condition image generation on stroke geometry, enabling direct control by user strokes with style codes corresponding to interactive brush styles. The generator uses an Alpha-color (AC) parametrization that outputs three colors and three spatial alpha maps instead of RGB images, enabling recoloring, compositing, and geometric control.

**Output:** Returns a stylized `PIL.Image` with applied brushstroke effects. The method supports patch-based rendering for canvases of any size and provides 239 pre-trained brush styles organized into categories: playdoh, beads, ribbon, string, misc, pompoms, chain, paper, and lego.

**Others:**

- **Model variants**: `style1` and `style2` (default)
- **Intermediate results**: Rendered PNG geometry image, patch-based stitching crops, alpha-color decomposed representation
- **Limitations**: Requires GPU for efficient inference (CUDA recommended), model files must be downloaded separately (automatic download supported)

## Usage Code

The main interface is the `Stylizer` class.

```python
from sketchkit.stylization import Stylizer
from sketchkit.datasets import PhotoSketching

# Initialize stylizer with default settings
stylizer = Stylizer(method="brushstrokesengine")

# Load a sketch
dataset = PhotoSketching()
sketch = dataset[0]

# Stylize the sketch
stylized_image = stylizer.run(sketch, canvas_size=512)

# Save the result
stylized_image.save("stylized_output.png")
```

### Custom Style Selection

```python
from sketchkit.stylization import Stylizer

# Initialize with specific style
stylizer = Stylizer(
    method="brushstrokesengine",
    model_name="style2",  # or "style1"
    style_id="playdoh10",  # style ID from brush library
    auto_download=True,
)

# Stylize with custom canvas size
stylized_image = stylizer.run(sketch, canvas_size=1024)
```

### Stylize from PNG Image

```python
from sketchkit.stylization import Stylizer

stylizer = Stylizer(method="brushstrokesengine")

# Stylize an existing PNG geometry image
stylized_image = stylizer.run_from_png(
    "input_sketch.png",
    output_path="output_stylized.png",
    style_id="misc05",
)
```

### Manual Model Download

```python
from sketchkit.stylization.brushstrokesengine.models import download_models, check_models_available

# Download models to default location (uses S3, falls back to Google Drive)
download_models()

# Or specify custom location
download_models(models_dir="/path/to/models")

# Use Google Drive directly (skip S3)
download_models(use_s3=False)

# Check if models are available
if check_models_available("style2"):
    print("Models are ready!")
```

## Usage Notes

**Constructor**

- `Stylizer(method="brushstrokesengine", model_name="style2", style_id="playdoh10", models_dir=None, device="cuda", auto_download=True)`
- `method`: Stylization method to use. Currently supported: `"brushstrokesengine"` (default).
- `model_name`: Model variant to use. Options: `"style1"`, `"style2"` (default).
- `style_id`: Style ID for stylization. Default: `"playdoh10"`.
- `models_dir`: Custom models directory. If None, uses default cache location.
- `device`: Device to use. Default: `"cuda"`.
- `auto_download`: Automatically download models if not found. Default: `True`.

**Properties**

- `stylizer.model_name`: Get the current model name (`"style1"` or `"style2"`).
- `stylizer.style_id`: Get the current style ID.

**run() Method**

- `run(sketch, output_path=None, canvas_size=None, style_id=None, library_path=None, **kwargs)`
- `sketch`: Input vector sketch to stylize.
- `output_path`: Output file path. If None, uses temporary path.
- `canvas_size`: Canvas size for rendering. If None, uses sketch's actual dimensions.
- `style_id`: Style ID to use. Overrides instance default if provided.
- `library_path`: Custom brush library path. If None, uses default (supports `projected_training_styles.pkl`, `projected_novel_styles.pkl`, `clip_brushes.pkl`).
- Additional parameters:
  - `feature_blending_level` (int, default: 2): Feature blending level for patch stitching.
  - `color_mode` (str, default: "1"): Color mode. Options: `"1"` (single style), `"2"` (blended styles).
  - `crop_margin` (int, default: 10): Margin for cropping patches.
  - `on_white` (bool, default: True): Render on white background.
  - `render_mode` (str, default: "clear"): Rendering mode. Options: `"clear"` (stroke with alpha), `"full"` (complete GAN output), `"stroke"` (opaque stroke), `"canvas"` (canvas only).
  - `stitching_mode` (str, default: "all"): Patch stitching mode. Options: `"all"` (all patches), `"full"` (full coverage).
  - `style_id2` (str, optional): Second style ID for style blending.
  - `style_blend_alpha` (float, default: 0.5): Blending weight between `style_id` and `style_id2` (0.0 = first style, 1.0 = second style).
  - `no_uvs_mapping` (bool, default: False): Disable UVS mapping.
  - `encoder_checkpoint` (str, optional): Path to geometry encoder checkpoint (`weak.pt` or `strong.pt`). If None, auto-detects from models directory.

**run_from_png() Method**

- `run_from_png(png_path, output_path=None, style_id=None, library_path=None, **kwargs)`
- `png_path`: Path to input PNG geometry image (black strokes on white background).
- `output_path`: Output file path. If None, uses temporary path.
- All other parameters same as `run()` method.

**Style Blending**

The method supports blending two styles by providing both `style_id` and `style_id2`:

```python
stylized_image = stylizer.run(
    sketch,
    style_id="playdoh10",
    style_id2="misc05",
    style_blend_alpha=0.3,  # 30% misc05, 70% playdoh10
)
```

**Available Styles**

The brush library contains **239 pre-trained styles** organized into categories: **playdoh** (13 base styles + variants), **beads** (22 base styles + variants), **ribbon** (32 base styles + variants), **string** (24 base styles + variants), **misc** (12 base styles + variants), **pompoms** (4 base styles + variants), **chain** (3 base styles + variants), **paper** (7 base styles + variants), and **lego** (3 base styles + variants). Many styles have multiple variants (e.g., `_take2`, `_take3`, `_take4`) representing different variations of the same base style.

Common style IDs include: `playdoh10`, `misc05`, `beads03`, `ribbon15`, `string12`, `pompoms02`, `chain00`, `paper01`, `lego00`, and many more.