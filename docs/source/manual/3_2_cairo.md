# Cairo Renderer

The Cairo renderer module provides functionality to rasterize vector sketches into images using the Cairo backend. This is useful for visualizing sketch data, generating raster outputs for downstream tasks, and applying configurable rendering options such as randomized colors.

**Source**: `tests/test_cairo_renderer.py`

## Available Renderer

The module currently supports the following renderer:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.renderer.cairo_renderer.CairoRenderer
```


# Code

The main interface is the ```CairoRenderer``` class.

```
from sketchkit.datasets import DifferSketching
from sketchkit.renderer.cairo_renderer import CairoRenderer

# Load a sketch dataset
my_data = DifferSketching(skip_integrity_check=True)
print(len(my_data))

# Initialize the renderer
renderer = CairoRenderer()

# Render a sketch with randomized colors
raster_image = renderer.render(my_data[0], {"random_color": True})

# Save the rendered result
raster_image.save("test1111111.png")

```