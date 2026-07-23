# Colorization

The colorization module provides functionality to colorize vector sketches or raster line art using AI-driven conditional generation.

**Source**: `tests/unit/test_colorization.py`

## Available Methods

The module currently supports the following methods:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.colorization.controlnet_lineart.ControlNetLineartModel
   sketchkit.colorization.controlnet_scribble.ControlNetScribbleModel
   sketchkit.colorization.manga_ninja.MangaNinjaModel

```

### ControlNet Lineart

The method implements [Adding Conditional Control to Text-to-Image Diffusion Models
(ControlNet)](https://arxiv.org/abs/2302.05543) from ICCV 2023 with a Lineart condition (`control_v11p_sd15_lineart`). It is designed to colorize clean, distinct line drawings and preserves the structural integrity of the original lines.

**Output:** Returns a `PIL.Image` (RGB mode).

### ControlNet Scribble

The method implements [ControlNet](https://arxiv.org/abs/2302.05543) with a Scribble condition (`sd-controlnet-scribble`). It is more tolerant of rough, freehand sketches and focuses on interpreting the overall shape and gesture of the input.

**Output:** Returns a `PIL.Image` (RGB mode).

### MangaNinja

The method implements [MangaNinja: Line Art Colorization with Precise Reference Following](https://arxiv.org/abs/2501.04552) from CVPR 2025. Unlike the ControlNet-based methods that rely on text prompts, MangaNinja uses a **reference image** to guide the colorization of line art, achieving remarkable consistency with the reference in terms of color, shading, and style. It also supports optional **point control** for precise spatial colorization guidance.

Key features:
- **Reference-guided**: Colors are transferred from a reference image instead of a text prompt.
- **Point control**: Optionally specify matching points on the reference and line art for precise color placement.
- **Auto lineart extraction**: When `is_lineart=False`, the built-in lineart detector extracts line art from the input automatically.

**Model weights** are automatically downloaded on first use:
- MangaNinja-specific weights (4 files, ~8 GB) → `~/.sketchkit/weights/colorization/manga_ninja/`
- SD 1.5, CLIP, ControlNet base models → `~/.cache/huggingface/`

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
    sketch, prompt="vibrant colors, anime style, high quality", size=512
)
result_lineart.save("colorized_lineart.png")

# 2) ControlNet Scribble: best for rough sketches
scribble_colorizer = Colorizer(method="controlnet_scribble", device="cuda")
result_scribble = scribble_colorizer.run(
    sketch, prompt="oil painting style, sunset, masterpiece", size=512
)
result_scribble.save("colorized_scribble.png")

# 3) MangaNinja: reference-guided colorization
ref_image = Image.open("reference.png").convert("RGB")
lineart_image = Image.open("lineart.png").convert("RGB")

manga_colorizer = Colorizer(method="manga_ninja", device="cuda")
result_manga = manga_colorizer.run(
    lineart_image,
    reference_image=ref_image,
    size=512,
    is_lineart=True,
    guidance_scale_ref=9.0,
)
result_manga.save("colorized_manga.png")

# 3b) MangaNinja with point control for precise colorization
import torch, numpy as np

point_ref = torch.from_numpy(np.load("point_ref.npy")).unsqueeze(0).unsqueeze(0)
point_main = torch.from_numpy(np.load("point_main.npy")).unsqueeze(0).unsqueeze(0)

result_point = manga_colorizer.run(
    lineart_image,
    reference_image=ref_image,
    size=512,
    is_lineart=True,
    guidance_scale_ref=9.0,
    guidance_scale_point=15.0,
    point_ref=point_ref,
    point_main=point_main,
)
result_point.save("colorized_manga_point.png")
```

## Usage Notes

**Constructor**

- `Colorizer(method="controlnet_lineart", device="cuda", **kwargs)`
  - `method`: backend name. Supported values are `"controlnet_lineart"`, `"controlnet_scribble"`, and `"manga_ninja"`.
  - `device`: execution device, e.g., `"cuda"` or `"cpu"`.

**Run Method**

- `run(input_data, prompt="", size=None, input_size=None, reference_image=None, **kwargs)`
  - `input_data`: accepts a `Sketch` object, `PIL.Image`, or `numpy.ndarray`.
  - `prompt`: text description guiding the colorization (default: `"vibrant colors, anime style"`). Not used by `manga_ninja`.
  - `size`: output resolution. Accepts an integer (square) or `(width, height)` tuple.
  - `input_size`: (Optional) resize dimensions for the input before processing.
  - `reference_image`: **Required for `manga_ninja`**. A `PIL.Image` or `numpy.ndarray` providing color reference.
  - `stroke_width` (int): Width of lines when rasterizing vector sketches (default: 2).
  - `num_inference_steps` (int): Diffusion steps (default: 20).
  - `guidance_scale` (float): Text prompt influence (default: 7.5). ControlNet methods only.
  - `controlnet_conditioning_scale` (float): Strength of the sketch control (default: 1.0). ControlNet methods only.
  - `seed` (int): Random seed for reproducibility.

**MangaNinja-specific Parameters**

- `is_lineart` (bool): Whether the input is already line art. If `False`, line art is automatically extracted (default: `False`).
- `guidance_scale_ref` (float): Strength of reference image guidance (default: 9.0).
- `guidance_scale_point` (float): Strength of point control guidance (default: 15.0).
- `point_ref` (torch.Tensor): Point map on the reference image, shape `(1, 1, H, W)`. Optional.
- `point_main` (torch.Tensor): Point map on the line art image, shape `(1, 1, H, W)`. Optional.