# Sketch2Image

The sketch2image module provides functionality to convert vector sketches into realistic images using various AI-driven image synthesis methods.

**Source**: `sketchkit/sketch2image/image_generator.py`

## Available Methods

The module currently supports the following methods:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.sketch2image.models.ControlNetModel
   sketchkit.sketch2image.models.Pix2PixModel
   sketchkit.sketch2image.models.T2IAdapterModel

```

### ControlNet

The method implements [Adding Conditional Control to Text-to-Image Diffusion Models
(ControlNet)](https://arxiv.org/abs/2302.05543) from ICCV 2023 using a scribble-based model (`sd-controlnet-scribble`). It tightly follows the spatial structure of the input sketch to generate a corresponding realistic image.

**Output:** Returns a `PIL.Image` (RGB mode).

### Pix2Pix

The method implements [InstructPix2Pix: Learning to Follow Image Editing Instructions
](https://arxiv.org/abs/2211.09800) (`instruct-pix2pix`) from CVPR 2023. It treats the generation as an image translation task, following instructions to convert the sketch context into a fully rendered image.

**Output:** Returns a `PIL.Image` (RGB mode).

### T2I-Adapter

The method implements [T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability for Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.08453) from AAAI 2024 using a Canny edge adapter (`t2iadapter_canny_sd15v2`). It provides a lightweight way to guide the Stable Diffusion generation using the edges derived from the sketch.

**Output:** Returns a `PIL.Image` (RGB mode).

## Code

The main interface is the `ImageGenerator` class.

```python
from sketchkit.sketch2image import ImageGenerator
from sketchkit.core import Sketch

# Load an input (Sketch object or Image)
# sketch = ... 

# 1) ControlNet: Scribble-conditioned generation
cnet_gen = ImageGenerator(method="controlnet", device="cuda")
img_cnet = cnet_gen.run(
    sketch, 
    prompt="a realistic photo of a cat", 
    size=512
)
img_cnet.save("cat_real_cnet.png")

# 2) Pix2Pix: Instruction-based translation
pix_gen = ImageGenerator(method="pix2pix", device="cuda")
img_pix = pix_gen.run(
    sketch, 
    prompt="turn this sketch into a realistic photo", 
    size=512,
    image_guidance_scale=1.2
)
img_pix.save("cat_real_pix2pix.png")

# 3) T2I-Adapter: Edge-guided generation
adapter_gen = ImageGenerator(method="t2i_adapter", device="cuda")
img_adapter = adapter_gen.run(
    sketch, 
    prompt="cinematic lighting, highly detailed", 
    size=512
)
img_adapter.save("cat_real_adapter.png")
```

## Usage Notes

**Constructor**

- `ImageGenerator(method="controlnet", device="cuda", **kwargs)`
  - `method`: backend name. Supported values are `"controlnet"`, `"pix2pix"`, and `"t2i_adapter"`.
  - `device`: execution device, e.g., `"cuda"` or `"cpu"`.

**Run Method**

- `run(input_data, prompt, size=None, input_size=None, **kwargs)`
  - `input_data`: accepts a `Sketch` object, `PIL.Image`, or `numpy.ndarray`.
  - `prompt`: text description guiding the generation (default: `"a realistic photo"`).
  - `size`: output resolution. Accepts an integer (square) or `(width, height)` tuple.
  - `stroke_width` (int): Width of lines when rasterizing vector sketches (default: 3).
  - `num_inference_steps` (int): Diffusion steps (varies by model, usually 20-50).
  - `guidance_scale` (float): Text prompt influence (default: 7.5).
  - `controlnet_conditioning_scale` (float): Strength of ControlNet (default: 1.0).
  - `adapter_conditioning_scale` (float): Strength of T2I-Adapter (default: 1.0).
  - `image_guidance_scale` (float): Strength of input image influence for Pix2Pix (default: 1.0).
  - `seed` (int): Random seed for reproducibility.