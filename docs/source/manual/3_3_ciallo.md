# Ciallo Renderer

The method implements [GPU-Accelerated Rendering of Vector Brush Strokes](https://dl.acm.org/doi/10.1145/3641519.3657418) from *SIGGRAPH 2024*. This is useful for high-quality sketch visualization, expressive brush-based rendering, and producing raster outputs with configurable canvas, stroke, brush texture, and background settings.

**Source**: `tests/test_ciallo_renderer.py`

## Available Renderer

The module currently supports the following renderer and render options:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.renderer.ciallo_renderer.CialloRenderer
   sketchkit.renderer.ciallo_renderer.CialloRenderOptions
```

# CialloRenderOptions

CialloRenderOptions stores configurable rendering parameters for CialloRenderer. These options can be used to control properties such as:

- brush texture path
- stroke color
- canvas size
- background color

# Code

The main interfaces are the ```CialloRenderer``` and ```CialloRenderOptions``` classes.

```
from sketchkit.datasets import DifferSketching
from sketchkit.renderer.ciallo_renderer import CialloRenderer, CialloRenderOptions

if __name__ == "__main__":
    my_data = DifferSketching(skip_integrity_check=True)
    print(len(my_data))
    
    render_options = CialloRenderOptions()
    renderer = CialloRenderer()

    render_options.canvas_size = (1024, 1024)
    render_options.brush_texture_path = "./assets/stamp1.png"
    render_options.background_color = (2, 0, 0, 1.0)
    render_options.stroke_color = (1, 0, 0, 255)

    img = renderer.render(my_data[0], render_options)
    img.save("ciallo_output_1111.png")
```