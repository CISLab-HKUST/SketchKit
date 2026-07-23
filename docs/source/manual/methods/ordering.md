# Stroke Ordering

The stroke ordering module determines the stroke drawing sequence for a static sketch, enabling applications like sketch animation and tutorials.

**Source**: `tests/unit/test_ordering_Fu.py`, `tests/unit/test_ordering_LineDrawer.py`

## Available Methods

The module currently supports the following methods:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.ordering.Fu
   sketchkit.ordering.LineDrawer
```

### Fu

The method implements [Animated construction of line drawings](https://dl.acm.org/doi/10.1145/2070781.2024167) from _ACM Transactions on Graphics (TOG) 2011_. It analyzes stroke relationships, computes transition costs between strokes, and determines the optimal drawing order using a branch-and-bound algorithm.

**Output:** Returns a single `Sketch` object with ordered paths, where the order of `sketch.paths` corresponds to the optimized drawing sequence.

### LineDrawer

The method implements [LineDrawer: Stroke-Level Process Reconstruction of Complex Line Art Based on Human Perception](https://www.sciencedirect.com/science/article/pii/S0097849325002067) from _Computers & Graphics 2025_. It uses a reference video (e.g., from [Paints-UNDO](https://github.com/lllyasviel/Paints-UNDO)) to recover the drawing order of a vector sketch by matching static vector strokes to the video frames where they first appear.

**Output:** Returns a single `Sketch` object with ordered paths, where the order of `sketch.paths` corresponds to the drawing sequence.

## Code

The main interface is the `Orderer` class.

```python
from sketchkit.ordering import Orderer
from sketchkit.core.sketch import Sketch
from PIL import Image
import numpy as np

# 1) Fu
orderer = Orderer(method="Fu", device="cpu")

# Run ordering using a Sketch object
from sketchkit.datasets import OpenSketch

dataset = OpenSketch()
sketch = dataset[0]
sketch = orderer.run(sketch)

# 2) LineDrawer
orderer = Orderer(method="LineDrawer", device="cuda")

# Run ordering using sketch image and reference video
img_path = "a1-2.png"
img_np = np.array(Image.open(img_path).convert("L"))
video_path = "a1-2.png.mp4"
sketch = orderer.run(img_np, video_path)

# Example: Access the paths
print(f"Generated ordered sketch with {sketch.path_num} paths.")
```

## Usage Notes

**Constructor**

- `Orderer(method=..., device=...)`
  - `method`: backend name. Supported values are `"Fu"`, `"LineDrawer"` (case-sensitive).
  - `device`: device to use for computation. For Fu, `"cpu"` is sufficient. For LineDrawer, `"cuda"` is recommended for better performance.

**Run Method**

- **Fu**: `run(sketch)`
  - `sketch`: `Sketch` object containing the static sketch to animate.

- **LineDrawer**: `run(img, ref_video_path)`
  - `img`: numpy array of the sketch image (grayscale).
  - `ref_video_path`: path to the reference video file. A reference video is required as input. See notes below for how two make it.

## Other Notes

**Steps of generating a reference video for LineDrawer:**

1.  Deploy Gradio interface of [Paints-UNDO](https://github.com/lllyasviel/Paints-UNDO?tab=readme-ov-file#get-started) according to its guidance.
2.  Click "Generate Key Frames".
3.  Set the step parameter to `700`.
4.  Generate `30` keyframes.
5.  Save as an `.mp4` file.
