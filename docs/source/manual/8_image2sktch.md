# Image2Sketch

The image2sketch module provides functionality to convert raster images into sketch-style outputs, including both raster edge maps and vector sketches.

**Source**: `tests/test_image2sketch.py`


## Available Methods

The module currently supports the following methods:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.image2sketch.SketchGenerator
   sketchkit.image2sketch.HEDModel
   sketchkit.image2sketch.PhotoSketchModel
   sketchkit.image2sketch.SwiftSketchModel
```

### HED

The method implements [Holistically-Nested Edge Detection](https://arxiv.org/abs/1504.06375) from *ICCV 2015*. It predicts dense edge maps from photos and optionally applies post-processing (non-maximum suppression and thinning) for cleaner sketch-like lines.

**Output:** Returns a grayscale `PIL.Image` (`"L"` mode). In the example below, the saved file is `angel_hed.png`.

### PhotoSketch

The method implements [Photo-Sketching: Inferring Contour Drawings from Images](https://arxiv.org/abs/1901.00542) from *WACV 2019*. It uses a Pix2Pix generator to map natural photos to contour-like sketch images.

**Output:** Returns a grayscale `PIL.Image` (`"L"` mode). In the example below, the saved file is `angel_photosketch.png`.

### SwiftSketch

The method implements [SwiftSketch: A Diffusion Model for Image-to-Vector Sketch Generation](https://arxiv.org/abs/2502.08642) from *SIGGRAPH 2025*. It predicts Bézier control points and converts them into SketchKit's vector `Sketch` representation.

**Output:** Returns a `Sketch` object. In the example below, it is rendered and saved as `angel_swiftsketch.png`, and can also be exported to SVG.



## Code

The main interface is the `SketchGenerator` class.

```python
from sketchkit.core import Sketch
from sketchkit.image2sketch import SketchGenerator
from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions
from PIL import Image

# Load an input image
input_image = Image.open("angel.png").convert("RGB")

# 1) HED: raster edge map
hed_generator = SketchGenerator(method="HED", postprocess=True)
hed_result = hed_generator.run(input_image, size=512)
hed_result.save("angel_hed.png")

# 2) PhotoSketch: raster contour sketch
photosketch_generator = SketchGenerator(method="PhotoSketch", device="cuda")
photosketch_result = photosketch_generator.run(input_image, size=512)
photosketch_result.save("angel_photosketch.png")

# 3) SwiftSketch: vector sketch
swift_model_kwargs = {
    "use_refine": True,
    "guidance_param": 2.5,
}
swiftsketch_generator = SketchGenerator(
    method="SwiftSketch",
    **swift_model_kwargs,
)
sketch = swiftsketch_generator.run(input_image, size=512)
assert isinstance(sketch, Sketch)

render_options = CairoRenderOptions(canvas_size=(512, 512))
renderer = CairoRenderer(render_options=render_options)
rendered_image = renderer.render(sketch, render_options=render_options)
rendered_image.save("angel_swiftsketch.png")

# Optional: save vector output
sketch.to_svg("angel_swiftsketch.svg")
```

- `SketchGenerator` initialization parameters:
  - **method**: backend name, one of `"HED"`, `"PhotoSketch"`, `"SwiftSketch"` (lowercase names are also supported).
  - **model_kwargs**: method-specific arguments, defined at initialization.
  - HED examples: `postprocess`, `postprocess_threshold`, `auto_download`, `checkpoint_dir`.
  - PhotoSketch examples: `device`, `auto_download`, `checkpoint_dir`.
  - SwiftSketch examples: `device`, `use_refine`, `guidance_param`, `auto_download`, `checkpoint_dir`.
- `SketchGenerator.run(...)` parameters:
  - **input**: input image (`PIL.Image`, `numpy.ndarray`, or image path).
  - **size**: output size. Use an integer for square output (e.g., `512`) or `(width, height)`. If omitted, output keeps the input image size.
- Advanced note:
  - `PhotoSketch` supports an optional init arg `input_size` (default `256`) for internal preprocessing resolution. `run(..., size=...)` controls output resolution.
  - `SwiftSketch` supports an optional init arg `fix_scale` (default `True`) for internal scale normalization.
