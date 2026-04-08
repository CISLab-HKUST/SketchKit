# Non-Photorealistic Renderer

The non-photorealistic renderer (NPR) module provides functionality to render 3D models into 2D sketch-like images using various NPR techniques such as suggestive contours, ridges and valleys, and apparent ridges.

**Source**: `sketchkit/renderer/npr_renderer.py`

## Available Renderer

The module currently supports the following renderer and render options:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.renderer.npr_renderer.NPRRenderer
   sketchkit.renderer.npr_renderer.NPRRenderOptions
```

### NPRRenderOptions

NPRRenderOptions stores configurable rendering parameters for NPRRenderer. These options can be used to control properties such as:

- canvas size
- rendering method
- threshold value for different rendering techniques

### NPRRenderer

The method implements non-photorealistic rendering techniques based on [Suggestive Contours for Conveying Shape](https://dl.acm.org/doi/10.1145/882262.882354) from *SIGGRAPH 2003*. It renders 3D models into 2D sketch-like images using various contour-based techniques.

**Output:** Returns a list of PIL Image objects representing the rendered views from different camera angles.

## Supported Rendering Methods

### Suggestive Contour

Suggestive contours are curves that appear when the surface bends away from the viewer, helping to convey the 3D shape of the model. This method uses a threshold parameter to control the sensitivity of contour detection.

### Ridges and Valleys

Ridges and valleys are curves that represent the highest and lowest points on the surface, respectively. This method highlights these features to emphasize the model's shape and structure.

### Apparent Ridges

Apparent ridges are view-dependent features that capture the perceived edges of the model, providing a more natural and intuitive representation of the 3D shape.

## Code

The main interfaces are the `NPRRenderer` and `NPRRenderOptions` classes.

```python
from sketchkit.renderer import NPRRenderer, NPRRenderOptions
from sketchkit.core.camera import Camera
import math
from sketchkit.utils.file import save_seq_gif

# Initialize render options
sug_opt = NPRRenderOptions(method="SuggestiveContour", threshold=0.005)
rv_opt = NPRRenderOptions(method="RidgesAndValleys", threshold=0.1)
ar_opt = NPRRenderOptions(method="AppearantRidges", threshold=0.1)

# Initialize renderer with 3D model
renderer = NPRRenderer(obj_path="media/rapid.obj")

# Create cameras at different angles
cameras = []
for angle in range(0, 360, 30):
    rad = math.radians(angle)
    camera = Camera()
    camera.set_look_at([0, 0, 0])
    camera.set_xyz([2.5 * math.cos(rad), 2.5 * math.sin(rad), 0])
    cameras.append(camera)

# Render with different methods
images = renderer._render(cameras, sug_opt)
save_seq_gif(images, "npr_rendering_suggestive_contour.gif")

images = renderer._render(cameras, rv_opt)
save_seq_gif(images, "npr_rendering_ridges_and_valleys.gif")

images = renderer._render(cameras, ar_opt)
save_seq_gif(images, "npr_rendering_appearant_ridges.gif")
```

## Usage Notes

**Constructor**

- `NPRRenderer(obj_path, render_options=None)`
- `obj_path`: path to the 3D model file (OBJ format)
- `render_options`: optional `NPRRenderOptions` object with rendering parameters

**NPRRenderOptions**

- `canvas_size`: tuple of (width, height) for the output images (default: (512, 512))
- `method`: rendering method to use. Supported values are `"SuggestiveContour"`, `"RidgesAndValleys"`, `"AppearantRidges"`
- `threshold`: threshold value for the selected rendering method (default: 0.005)

**Render Method**

- `_render(cameras, render_options=None)`
  - `cameras`: list of `Camera` objects defining different viewpoints
  - `render_options`: optional rendering options (overrides the renderer's default options)
  - Returns: list of PIL Image objects

**Normalization**

The renderer automatically normalizes the 3D model to fit within a unit cube, which helps ensure consistent rendering results across different models.