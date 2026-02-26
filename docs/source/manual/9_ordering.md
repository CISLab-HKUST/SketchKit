# Ordering

The ordering module determines the stroke drawing sequence for a static sketch, enabling applications like sketch animation and tutorials.

## Overview

The ordering module contains algorithms that determine the optimal drawing order of strokes in a sketch. This is particularly useful for creating animated sketches or generating step-by-step drawing tutorials. There are two main approaches implemented:

1. **LineDrawer**: Uses reference videos to determine stroke order
2. **Fu**: Uses optimization algorithms to determine stroke order

## LineDrawer

LineDrawer uses a reference video (e.g., from [Paints-UNDO](https://github.com/lllyasviel/Paints-UNDO)) to recover the drawing order of a vector sketch. It matches static vector strokes to the video frames where they first appear.

### Generate Key Frames

To create the reference video:
1.  Open **Paints-UNDO**.
2.  Go to the **Generate Key Frames** step.
3.  Set the step parameter to **[700]**.
4.  Generate **30** keyframes.
5.  Save as an `.mp4` file.

### Usage Example

```python
from sketchkit.ordering import Orderer
from sketchkit.core.sketch import Sketch
from PIL import Image
import numpy as np

# Initialize Orderer with LineDrawer method
orderer = Orderer.create(method="LineDrawer", device="cuda")

# Run ordering using sketch image and reference video
# img_path: Static sketch path
# video_path: Reference video path
img_path = "a1-2.png"
img_np = np.array(Image.open(img_path).convert("L"))
video_path = "a1-2.png.mp4"

# Returns a single Sketch object with ordered paths
sketch = orderer.run(img_np, video_path)

# Example: Access the paths
print(f"Generated ordered sketch with {sketch.path_num} paths.")
```

The output is a single `Sketch` object where the order of `sketch.paths` corresponds to the drawing sequence.

## Fu Algorithm

The Fu algorithm is an optimization-based approach that determines the drawing order by solving a traveling salesman problem variant. It considers costs associated with individual strokes and transitions between strokes to find an optimal drawing sequence.

### Parameters

The Fu algorithm accepts the following parameters:
- `max_k`: Maximum number of neighbors to consider for each stroke (default: 4)
- `w`: Weight factor for transition costs (default: 0.1111111)
- `canvas_size`: Size of the canvas for rendering (default: 800)
- `canvas_color`: Background color of the canvas (default: white `(1, 1, 1)`)
- `width`: Width of the strokes (default: 4)

### Usage Example

```python
from sketchkit.ordering import Orderer
from sketchkit.datasets import QuickDraw
from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions
from sketchkit.utils.file import save_seq_gif
from tqdm import tqdm

# Load a sketch from dataset
my_data = QuickDraw(cislab_source=True)
sketch = my_data[10]

# Initialize Orderer with Fu method
animated_ordering = Orderer.create(method="Fu", device="cpu")

# Run ordering on the sketch
ordered_sketch = animated_ordering.run(sketch)

# Generate progressive frames for animation
sketch_frames = ordered_sketch.get_progressive_frames(length_per_frame=10)

# Render each frame
render_options = CairoRenderOptions(
    canvas_size=(800, 800),
    background_color=(1.0, 1.0, 1.0)
)
renderer = CairoRenderer(render_options)

raster_images = [renderer.render(frame) for frame in tqdm(sketch_frames)]

# Save individual frames
for i, raster_image in enumerate(raster_images):
    raster_image.save(f'media/frames/frame_{i:04d}.png', 'PNG')

# Save as animated GIF
save_seq_gif(raster_images, 'animated_drawing.gif')

# Example: Save the ordered sketch
ordered_sketch.to_svg(filename='ordered_sketch.svg')
```

## Orderer Class

The `Orderer` class is an abstract base class that provides a unified interface for different ordering methods. It supports factory creation of different ordering implementations.

### Factory Method

Use the static `create` method to instantiate different ordering implementations:

```python
from sketchkit.ordering import Orderer

# Create LineDrawer instance
line_drawer = Orderer.create(method="LineDrawer", device="cuda")

# Create Fu instance  
fu_orderer = Orderer.create(method="Fu", device="cpu")
```

### Abstract Interface

All ordering implementations must implement the `run` method which takes input data and returns an ordered `Sketch` object.

The output is a single `Sketch` object where the order of `sketch.paths` corresponds to the drawing sequence.