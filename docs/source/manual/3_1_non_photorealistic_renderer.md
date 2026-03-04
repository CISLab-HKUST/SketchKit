# Non-Photorealistic Renderer (NPR)

The Non-Photorealistic Renderer (NPR) module provides functionality to render 3D geometric models using various artistic line drawing techniques, emphasizing shape and form through stylized representations rather than realistic shading.

**Source**: `renderer/npr_renderer.py`

## Overview

The NPR renderer transforms 3D models (in OBJ format) into 2D line drawings using specialized algorithms that highlight geometric features. It supports multiple rendering techniques and camera viewpoints to generate sequences of images for animation or analysis.

Key features include:
- Automatic model normalization to ensure consistent results
- Multiple rendering techniques for different artistic effects
- Support for custom camera viewpoints and animation
- Integration with SketchKit's camera and image utilities

## NPRRenderOptions

The `NPRRenderOptions` data class defines the configuration parameters for the NPR renderer:

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `canvas_size` | `tuple[int, int]` | `(512, 512)` | Output image dimensions (width, height) |
| `method` | `str` | `"SuggestiveContour"` | Rendering technique to use |
| `threshold` | `float` | `0.005` | Sensitivity threshold for feature detection |

### Supported Rendering Methods

- `SuggestiveContour`: Highlights contours that suggest the presence of curved surfaces
- `RidgesAndValleys`: Emphasizes geometric features like creases, ridges, and valleys
- `AppearantRidges`: Shows ridges that are visually prominent from specific viewpoints

## NPRRenderer

The `NPRRenderer` class handles the loading of 3D models and the actual rendering process.

### Initialization

```python
from sketchkit.renderer.npr_renderer import NPRRenderer, NPRRenderOptions

# Basic initialization with default options
renderer = NPRRenderer(obj_path="model.obj")

# Custom initialization with specific options
render_options = NPRRenderOptions(
    canvas_size=(800, 600),
    method="SuggestiveContour",
    threshold=0.01
)
renderer = NPRRenderer(obj_path="model.obj", render_options=render_options)
```

### Model Normalization

The renderer automatically normalizes the input OBJ model to fit within a unit cube and centers it at the origin. This preprocessing step ensures consistent rendering results regardless of the original model's scale or position.

### Rendering Process

The `_render` method generates 2D images from specified camera viewpoints:

```python
from sketchkit.renderer.npr_renderer import NPRRenderer
from sketchkit.core.camera import Camera
import math

# Initialize renderer
renderer = NPRRenderer(obj_path="model.obj")

# Create camera viewpoints
cameras = []
for angle in range(0, 360, 30):
    camera = Camera()
    camera.set_look_at([0, 0, 0])
    camera.set_xyz([2.5 * math.cos(math.radians(angle)), 2.5 * math.sin(math.radians(angle)), 0])
    cameras.append(camera)

# Render images
images = renderer._render(cameras)
```

### Rendering Techniques

#### Suggestive Contour

Emphasizes contours that suggest the presence of curved surfaces:

```python
from sketchkit.renderer.npr_renderer import NPRRenderOptions

# Create render options
suggestive_options = NPRRenderOptions(
    method="SuggestiveContour",
    threshold=0.005
)

# Render with these options
images = renderer._render(cameras, suggestive_options)
```

#### Ridges and Valleys

Highlights geometric features like creases, ridges, and valleys:

```python
ridges_options = NPRRenderOptions(
    method="RidgesAndValleys",
    threshold=0.1
)
images = renderer._render(cameras, ridges_options)
```

#### Apparent Ridges

Shows ridges that are visually prominent from specific viewpoints:

```python
apparent_options = NPRRenderOptions(
    method="AppearantRidges",
    threshold=0.1
)
images = renderer._render(cameras, apparent_options)
```

## Complete Usage Example

```python
from sketchkit.renderer.npr_renderer import NPRRenderer, NPRRenderOptions
from sketchkit.core.camera import Camera
from sketchkit.utils.file import save_seq_gif
import math

# Define render options for different techniques
techniques = [
    ("SuggestiveContour", 0.005, "suggestive_contour.gif"),
    ("RidgesAndValleys", 0.1, "ridges_and_valleys.gif"),
    ("AppearantRidges", 0.1, "apparent_ridges.gif")
]

# Initialize renderer
renderer = NPRRenderer(obj_path="media/model.obj")

# Create multiple camera viewpoints for animation
cameras = []
for angle in range(0, 360, 15):  # 24 viewpoints
    rad = math.radians(angle)
    camera = Camera()
    camera.set_look_at([0, 0, 0])
    camera.set_xyz([2.5 * math.cos(rad), 0.5, 2.5 * math.sin(rad)])  # Slightly above center
    cameras.append(camera)

# Render and save animations for each technique
for method, threshold, filename in techniques:
    render_options = NPRRenderOptions(
        canvas_size=(800, 800),
        method=method,
        threshold=threshold
    )
    
    # Render images
    print(f"Rendering {method}...")
    images = renderer._render(cameras, render_options)
    
    # Save as animated GIF
    save_seq_gif(images, filename)
    print(f"Saved animation to {filename}")
```

## External Dependencies

The NPR renderer relies on an external executable called `nogui_contour_renderer` which must be present in the renderer module directory. This executable performs the actual rendering calculations and generates PPM image files that are then loaded and processed by the Python wrapper.

The renderer also uses:
- `trimesh` library for OBJ model loading and manipulation
- `PIL` for image processing
- `numpy` for mathematical operations
- `tempfile` for temporary file management