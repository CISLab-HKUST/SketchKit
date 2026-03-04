# Colorization

The colorization module provides functionality to colorize vector sketches or raster line art using AI-driven conditional generation.

**Source**: `sketchkit/colorization/colorizer.py`

## Available Methods

The module currently supports the following methods:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.colorization.models.ControlNetLineartModel
   sketchkit.colorization.models.ControlNetScribbleModel

```

### ControlNet Lineart

The method implements [ControlNet](https://arxiv.org/abs/2302.05543) with a Lineart condition (`control_v11p_sd15_lineart`). It is designed to colorize clean, distinct line drawings and preserves the structural integrity of the original lines.

**Output:** Returns a `PIL.Image` (RGB mode).

### ControlNet Scribble

The method implements [ControlNet](https://arxiv.org/abs/2302.05543) with a Scribble condition (`sd-controlnet-scribble`). It is more tolerant of rough, freehand sketches and focuses on interpreting the overall shape and gesture of the input.

**Output:** Returns a `PIL.Image` (RGB mode).

## Code

The main interface is the `Colorizer` class.

```python
from sketchkit.colorization import Colorizer
from sketchkit.core import Sketch
from PIL import Image

# Load an input (Sketch object or Image)
# Assuming 'sketch' is a loaded SketchKit Sketch object
# input_image = Image.open("sketch.png").convert("RGB")

# 1) ControlNet Lineart: best for clean lines
lineart_colorizer = Colorizer(method="controlnet_lineart", device="cuda")
result_lineart = lineart_colorizer.run(
    sketch, 
    prompt="vibrant colors, anime style, high quality", 
    size=512
)
result_lineart.save("colorized_lineart.png")

# 2) ControlNet Scribble: best for rough sketches
scribble_colorizer = Colorizer(method="controlnet_scribble", device="cuda")
result_scribble = scribble_colorizer.run(
    sketch, 
    prompt="oil painting style, sunset, masterpiece", 
    size=512
)
result_scribble.save("colorized_scribble.png")

```

## Usage Notes

**Constructor**

* `Colorizer(method="controlnet_lineart", device="cuda", **kwargs)`
* `method`: backend name. Supported values are `"controlnet_lineart"` and `"controlnet_scribble"`.
* `device`: execution device, e.g., `"cuda"` or `"cpu"`.

**Run Method**

* `run(input_data, prompt, size=None, input_size=None, **kwargs)`
* `input_data`: accepts a `Sketch` object, `PIL.Image`, or `numpy.ndarray`.
* `prompt`: text description guiding the colorization (default: `"vibrant colors, anime style"`).
* `size`: output resolution. Accepts an integer (square) or `(width, height)` tuple.
* `input_size`: (Optional) resize dimensions for the input before processing.
* `**kwargs`: Generation controls including:
* `stroke_width` (int): Width of lines when rasterizing vector sketches (default: 2).
* `num_inference_steps` (int): Diffusion steps (default: 20).
* `guidance_scale` (float): Text prompt influence (default: 7.5).
* `controlnet_conditioning_scale` (float): Strength of the sketch control (default: 1.0).
* `seed` (int): Random seed for reproducibility.