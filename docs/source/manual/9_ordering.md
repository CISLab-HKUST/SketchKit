# Ordering

The ordering module determines the stroke drawing sequence for a static sketch, enabling applications like sketch animation and tutorials.

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
orderer = Orderer(order_method="LineDrawer", device="cuda")

# Run ordering using sketch image and reference video
# img_path: Static sketch path
# video_path: Reference video path
img_path="a1-2.png"
img_np = np.array(Image.open(img_path).convert("L"))
video_path="a1-2.png.mp4"

# Returns a single Sketch object with ordered paths
sketch = orderer.run(img_np, video_path)

# Example: Access the paths
print(f"Generated ordered sketch with {sketch.path_num} paths.")
```

The output is a single `Sketch` object where the order of `sketch.paths` corresponds to the drawing sequence.
