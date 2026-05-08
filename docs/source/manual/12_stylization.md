# Stylization

The stylization module provides functionality to convert vector sketches or geometry images into stylized raster images with learned brushstroke effects.

**Source**: `tests/unit/test_stylizer.py`

## Available Methods

The module currently supports the following interface:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.stylization.NeuralBrushstroke
```

### NeuralBrushstroke

The backend implements [Neural Brushstroke Engine: Learning a Latent Space of Interactive Drawing Tools](https://research.nvidia.com/labs/toronto-ai/brushstroke_engine/) from *SIGGRAPH Asia 2022*. It stylizes clean sketch geometry with learned brushstroke appearance while preserving the input structure.

**Output:** Returns a stylized `PIL.Image`. The backend supports multiple pretrained styles, style blending, and both `Sketch` and PNG geometry inputs. Required checkpoints are downloaded automatically when missing during initialization.

## Code

The main interface is the `Stylizer` class.

```python
from sketchkit.datasets import PhotoSketching
from sketchkit.stylization import Stylizer

dataset = PhotoSketching()
sketch = dataset[0]

stylizer = Stylizer(
    method="NeuralBrushstroke",
    model_name="style2",
    style_id="playdoh10",
)

# 1) Stylize from a Sketch
stylized_image = stylizer.run(
    sketch,
    output_path="stylized_output.png",
    canvas_size=512,
)

# 2) Stylize from a PNG geometry image
png_result = stylizer.run_from_png(
    "input_sketch.png",
    output_path="stylized_from_png.png",
    style_id="misc05",
)

# 3) Blend two styles
blend_result = stylizer.run(
    sketch,
    output_path="stylized_blend.png",
    style_id="playdoh10",
    style_id2="misc05",
    style_blend_alpha=0.3,
)
```

## Arguments

**Constructor**

- `method`: currently supports `"NeuralBrushstroke"`.
- `model_name`: pretrained model variant. Supported values are `"style1"` and `"style2"`.
- `style_id`: default style used for stylization.

**Run Methods**

- `canvas_size`: square rasterization size for `run(sketch, ...)`. If omitted, the sketch's own dimensions are used.
- `style_id`: optional per-call style override for `run()` or `run_from_png()`.
- `style_id2`: optional second style for blending.
- `style_blend_alpha`: blend weight between `style_id` and `style_id2`.
- `png_path`: input geometry image for `run_from_png(...)`; it should be a black-on-white line drawing.
