# Get Started

## Install uv

SketchKit uses [`uv`](https://docs.astral.sh/uv/) to manage Python dependencies and the project environment. Install `uv` before installing SketchKit.

**macOS and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If `curl` is not available, you can use `wget` instead:

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

**Windows PowerShell:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, restart your terminal or reload your shell, then verify that `uv` is available:

```bash
uv --version
```


## Install SketchKit

#### Step 1: Prepare SketchKit Files

Extract `SketchKit.zip` to a folder named `SketchKit`

#### Step 2: Open Terminal

Navigate to the `SketchKit` folder in your terminal:

**Method 1 (File Manager):**

- Right-click in the `SketchKit` folder and select "Open in Terminal" or "Open PowerShell here"

**Method 2 (Command Line):**

```bash
cd /path/to/SketchKit
```

Replace `/path/to/SketchKit` with the actual path to your SketchKit folder.

#### Step 3: Install SketchKit

Run the following commands in the working directory:

```bash
# Install the environment
uv sync

# Activate the environment
source .venv/bin/activate
```

After installation, you can import `sketchkit` in your python program.

#### Step 4: Verify Installation

Test your installation:

```python
import sketchkit
from sketchkit.datasets import OpenSketch
print("SketchKit installed successfully!")
```

**Note**: Use `pip install -e .` (with `-e`) for development installation, which allows you to modify the code and see changes immediately.

## Examples

### Example 1: Dataset Loading and Rendering

This example shows how to load a sketch from a dataset and render it into a raster image.

#### Pipeline

1. Load a sketch dataset.
2. Get one sketch sample from the dataset.
3. Render the sketch into a raster image.
4. Save the rendered result as a PNG file.

#### Code

```python
from sketchkit.datasets import OpenSketch
from sketchkit.renderer import CairoRenderer


if __name__ == "__main__":
    # Load a sketch dataset
    dataset = OpenSketch()

    # Get the first sketch from the dataset
    sketch = dataset[0]

    # Initialize the renderer
    renderer = CairoRenderer()

    # Render the vector sketch into a raster image
    raster_image = renderer.render(sketch)

    # Save the rendered result
    raster_image.save("rendering.png")

    print("Rendered image saved to: rendering.png")
```

#### Output

After running this script, you should see an output image: `rendering.png`. This image is the rendered raster version of the loaded sketch.

### Example 2: Sketch Vectorization and Stroke Ordering

This example shows how to convert a raster sketch into a vector sketch and generate a stroke-by-stroke drawing animation.

#### Pipeline

1. Load an input raster sketch image.
2. Convert the raster sketch into a vector sketch.
3. Estimate a plausible stroke drawing order.
4. Generate progressive drawing frames.
5. Save the drawing process as a GIF.

#### Code

```python
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from sketchkit.ordering import Orderer
from sketchkit.renderer.cairo_renderer import CairoRenderer
from sketchkit.utils.file import save_seq_gif
from sketchkit.vectorization import Vectorizer


if __name__ == "__main__":
    # Input raster sketch
    img_path = "test_files/butterfly.png"
    img_np = np.array(Image.open(img_path).convert("L"))

    # Prepare output folders
    output_path = "outputs/test"
    output_images_path = os.path.join(output_path, "frames")
    output_gif_path = os.path.join(output_path, "sketch.gif")

    os.makedirs(output_images_path, exist_ok=True)

    # Step 1: Vectorize the raster sketch
    vectorizer = Vectorizer(method="DeepVecSIG24")
    sketch = vectorizer.run(img_np)

    # Step 2: Estimate stroke drawing order
    orderer = Orderer(method="Fu", device="cpu")
    sketch = orderer.run(sketch)

    # Step 3: Generate progressive sketch frames
    sketch_frames = sketch.get_progressive_frames(length_per_frame=10)

    # Step 4: Render each frame
    renderer = CairoRenderer()
    raster_images = [renderer.render(frame) for frame in tqdm(sketch_frames)]

    # Step 5: Save frames as PNG images
    for i, raster_image in enumerate(raster_images):
        raster_image.save(
            os.path.join(output_images_path, f"frame_{i:04d}.png"),
            "PNG",
        )

    # Step 6: Save the drawing process as a GIF
    save_seq_gif(raster_images, output_gif_path)

    print(f"Drawing animation saved to: {output_gif_path}")
```

#### Output

After running this script, the generated files will be saved to:

```
outputs/test/
├── frames/
│ ├── frame_0000.png
│ ├── frame_0001.png
│ └── ...
└── sketch.gif
```

The file sketch.gif shows the sketch being drawn progressively stroke by stroke.

### Example 3: NPR Rendering, Vectorization, and Stylization

This example demonstrates a complete pipeline that combines non-photorealistic rendering (NPR), vectorization, and stylization.

Starting from a 3D mesh model, SketchKit first renders sketch-like images from multiple camera viewpoints using an NPR renderer. Then, one rendered raster sketch is converted into a vector sketch. Finally, the vector sketch is stylized into an artistic output image.

#### Pipeline

1. Render sketch-like images from a 3D object using the NPR renderer.
2. Save the rendered sequence as a GIF.
3. Convert one rendered raster sketch into a vector sketch.
4. Export the vector sketch as an SVG file.
5. Stylize the vector sketch into a final raster image.

#### Code

```python
import math
import os

import numpy as np

from sketchkit.core.camera import Camera
from sketchkit.renderer.npr_renderer import NPRRenderer, NPRRenderOptions
from sketchkit.stylization import Stylizer
from sketchkit.utils.file import save_seq_gif
from sketchkit.vectorization import Vectorizer


if __name__ == "__main__":
    output_path = "outputs/test"
    os.makedirs(output_path, exist_ok=True)

    # Initialize the NPR renderer with a 3D mesh
    renderer = NPRRenderer(obj_path="test_files/head.obj")

    # Create a set of cameras rotating around the object
    cameras = []
    radius = 2.5
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        camera = Camera()
        camera.set_look_at([0, 0, 0])
        camera.set_xyz([radius * math.cos(rad), radius * math.sin(rad), 0])
        cameras.append(camera)

    print(f"Starting render for {len(cameras)} cameras...")

    # Render sketch-like images using the Suggestive Contour method
    print("Rendering with SuggestiveContour method...")
    options = NPRRenderOptions(method="SuggestiveContour", threshold=0.005)
    images = renderer._render(cameras, options)

    # Save the rendering sequence as a GIF
    gif_path = os.path.join(output_path, "SuggestiveContour.gif")
    save_seq_gif(images, gif_path)

    # Save the first rendered image as a raster sketch
    raster_sketch_path = os.path.join(output_path, "raster_sketch.png")
    images[0].save(raster_sketch_path)

    # Convert the raster sketch to a grayscale numpy array
    img_gray = images[0].convert("L")
    img_array = np.array(img_gray)

    # Vectorize the raster sketch
    vectorizer = Vectorizer(method="LineDrawer")
    sketch = vectorizer.run(img_array)

    # Export the vector sketch as an SVG file
    vector_sketch_path = os.path.join(output_path, "vector_sketch.svg")
    sketch.to_svg(filename=vector_sketch_path)

    # Stylize the vector sketch
    stylizer = Stylizer(
        method="NeuralBrushstroke",
        model_name="style2",
        style_id="playdoh10",
    )

    stylized_output_path = os.path.join(output_path, "stylized_sketch.png")
    stylized_image = stylizer.run(
        sketch,
        output_path=stylized_output_path,
        canvas_size=512,
    )

    print(f"Rendering GIF saved to: {gif_path}")
    print(f"Raster sketch saved to: {raster_sketch_path}")
    print(f"Vector sketch saved to: {vector_sketch_path}")
    print(f"Stylized image saved to: {stylized_output_path}")
```

#### Output

After running this example, the generated files will be saved to:

```
outputs/test/
├── SuggestiveContour.gif
├── raster_sketch.png
├── vector_sketch.svg
└── stylized_sketch.png
```
