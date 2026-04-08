# Renderer

The renderer module provides functionality to convert vector sketches into raster images for visualization. 

## Available Methods

The module currently supports the following renderers:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.renderer.CairoRenderer
   sketchkit.renderer.CialloRenderer
```

### Cairo Renderer

The Cairo renderer module provides functionality to rasterize vector sketches into images using the Cairo backend. This is useful for visualizing sketch data, generating raster outputs for downstream tasks, and applying configurable rendering options such as randomized colors.

**Source**: `tests/test_cairo_renderer.py`


#### Code

The main interface is the ```CairoRenderer``` class.

```python
from sketchkit.datasets import DifferSketching
from sketchkit.renderer import CairoRenderer

# Load a sketch dataset
my_data = DifferSketching(skip_integrity_check=True)
sketch = my_data[0]

# Initialize the renderer
renderer = CairoRenderer()

# Render a sketch into a raster image
raster_image = renderer.render(sketch)

# Save the rendered result
raster_image.save("cairo_rendering.png")
```

### Ciallo Renderer

The method implements [GPU-Accelerated Rendering of Vector Brush Strokes](https://dl.acm.org/doi/10.1145/3641519.3657418) from *SIGGRAPH 2024*. This is useful for high-quality sketch visualization, expressive brush-based rendering, and producing raster outputs with configurable canvas, stroke, brush texture, and background settings.

**Source**: `tests/test_ciallo_renderer.py`

#### CialloRenderOptions

CialloRenderOptions stores configurable rendering parameters for CialloRenderer. These options can be used to control properties such as:

- brush texture path
- stroke color
- canvas size
- background color

#### Code

The main interfaces are the ```CialloRenderer``` and ```CialloRenderOptions``` classes.

```python
from sketchkit.datasets import DifferSketching
from sketchkit.renderer import CialloRenderer, CialloRenderOptions

my_data = DifferSketching(skip_integrity_check=True)
sketch = my_data[0]

render_options = CialloRenderOptions()
renderer = CialloRenderer()

render_options.canvas_size = (1024, 1024)
render_options.brush_texture_path = "./assets/stamp1.png"
render_options.background_color = (2, 0, 0, 1.0)
render_options.stroke_color = (1, 0, 0, 255)

raster_image = renderer.render(sketch, render_options)
raster_image.save("ciallo_rendering.png")
```