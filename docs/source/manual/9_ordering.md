# Ordering

The ordering module determines the stroke drawing sequence for a static sketch, enabling applications like sketch animation and tutorials.

**Source**: `tests/animated_drawing_test.py`

## Available Methods

The module currently supports the following methods:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.ordering.LineDrawer
   sketchkit.ordering.Fu
```

### LineDrawer

The method implements [LineDrawer: Stroke-Level Process Reconstruction of Complex Line Art Based on Human Perception](https://www.sciencedirect.com/science/article/pii/S0097849325002067) from *Computers & Graphics 2025*. It uses a reference video (e.g., from [Paints-UNDO](https://github.com/lllyasviel/Paints-UNDO)) to recover the drawing order of a vector sketch by matching static vector strokes to the video frames where they first appear.

**Output:** Returns a single `Sketch` object with ordered paths, where the order of `sketch.paths` corresponds to the drawing sequence.

### Fu

The method implements [Animated construction of line drawings](https://dl.acm.org/doi/10.1145/2070781.2024167) from *ACM Transactions on Graphics (TOG) 2011*. It analyzes stroke relationships, computes transition costs between strokes, and determines the optimal drawing order using a branch-and-bound algorithm.

**Output:** Returns a single `Sketch` object with ordered paths, where the order of `sketch.paths` corresponds to the optimized drawing sequence.

## Code

The main interface is the `Orderer` class.

```python
from sketchkit.ordering import Orderer
from sketchkit.core.sketch import Sketch
from PIL import Image
import numpy as np

# 1) LineDrawer
orderer = Orderer(method="LineDrawer", device="cuda")

# Run ordering using sketch image and reference video
img_path = "a1-2.png"
img_np = np.array(Image.open(img_path).convert("L"))
video_path = "a1-2.png.mp4"
sketch = orderer.run(img_np, video_path)

# 2) Fu
orderer = Orderer(method="Fu", device="cpu")

# Run ordering using a Sketch object
from sketchkit.datasets import QuickDraw
dataset = QuickDraw(cislab_source=True)
sketch = dataset[0]
sketch = orderer.run(sketch)

# Example: Access the paths
print(f"Generated ordered sketch with {sketch.path_num} paths.")
```

## Usage Notes

**Constructor**

- `Orderer(method=..., device=...)`
- `method`: backend name. Supported values are `"LineDrawer"`, `"Fu"` (case-sensitive).
- `device`: device to use for computation. For LineDrawer, `"cuda"` is recommended for better performance; for Fu, `"cpu"` is sufficient.

**Run Method**

- **LineDrawer**: `run(img, ref_video_path)`
  - `img`: numpy array of the sketch image (grayscale)
  - `ref_video_path`: path to the reference video file

- **Fu**: `run(sketch)`
  - `sketch`: `Sketch` object containing the static sketch to animate

**Generate Key Frames for LineDrawer**

To create the reference video for LineDrawer:
1.  Open **Paints-UNDO**.
2.  Go to the **Generate Key Frames** step.
3.  Set the step parameter to **[700]**.
4.  Generate **30** keyframes.
5.  Save as an `.mp4` file.
