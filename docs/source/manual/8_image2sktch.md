# Image2Sketch

The image2sketch module provides functionality to convert raster images into sketch-style outputs, including both raster edge maps and vector sketches.

**Source**: `tests/test_image2sketch.py`


## Available Methods

The module currently supports the following methods:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.image2sketch.HEDModel
   sketchkit.image2sketch.PhotoSketchModel
   sketchkit.image2sketch.SwiftSketchModel
```

### HED

The method implements [Holistically-Nested Edge Detection](https://arxiv.org/abs/1504.06375). It predicts dense edge maps from photos and optionally applies post-processing (non-maximum suppression and thinning) for cleaner sketch-like lines.

**Output:** Returns a grayscale `PIL.Image` (`"L"` mode). In the example below, the saved file is `angel_hed.png`.

### PhotoSketch

The method implements [Photo-Sketching: Inferring Contour Drawings from Images](https://arxiv.org/abs/1901.00542). It uses a Pix2Pix generator to map natural photos to contour-like sketch images.

**Output:** Returns a grayscale `PIL.Image` (`"L"` mode). In the example below, the saved file is `angel_photosketch.png`.

### SwiftSketch

SwiftSketch is a diffusion-based image-to-vector-sketch method. It predicts Bézier control points and converts them into SketchKit's vector `Sketch` representation.

**Output:** Returns a `Sketch` object. In the example below, it is rendered and saved as `angel_swiftsketch.png`, and can also be exported to SVG.


## Code

The main interface is the `image2sketch` function.

```python
from sketchkit.core import Sketch
from sketchkit.image2sketch import image2sketch
from sketchkit.renderer.cairo_renderer import CairoRenderer
from PIL import Image

# Load an input image
input_image = Image.open("angel.png").convert("RGB")

# 1) HED: raster edge map
hed_result = image2sketch(input_image, method="hed", size=512)
hed_result.save("angel_hed.png")

# 2) PhotoSketch: raster contour sketch
photosketch_result = image2sketch(input_image, method="photosketch", size=512)
photosketch_result.save("angel_photosketch.png")

# 3) SwiftSketch: vector sketch
swiftsketch_result = image2sketch(input_image, method="swiftsketch", size=512)
assert isinstance(swiftsketch_result, Sketch)

renderer = CairoRenderer(canvas_size=512)
rendered_array = renderer.render(swiftsketch_result, fit_canvas=True)
rendered_image = Image.fromarray(rendered_array.astype("uint8")).convert("RGB")
rendered_image.save("angel_swiftsketch.png")

# Optional: save vector output
swiftsketch_result.to_svg("angel_swiftsketch.svg")
```

- Parameters:
  - **method**: backend name, one of `"hed"`, `"photosketch"`, `"swiftsketch"`.
  - **size**: output size. Use an integer for square output (e.g., `512`) or `(width, height)`.
  - **model_kwargs**: method-specific initialization arguments.
    - HED examples: `postprocess`, `postprocess_threshold`, `auto_download`, `checkpoint_dir`.
    - PhotoSketch examples: `device`, `input_size`, `auto_download`, `checkpoint_dir`.
    - SwiftSketch examples: `device`, `use_refine`, `guidance_param`, `fix_scale`, `auto_download`, `checkpoint_dir`.
