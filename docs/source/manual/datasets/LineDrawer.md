# LineDrawer

`LineDrawer` is a `SketchDataset` loader for the [LineDrawer](https://www.sciencedirect.com/science/article/pii/S0097849325002067) dataset, where the sequential drawing process of high-quality anime line arts is stored as `.json` files. The loader extracts coordinate points and dynamic point-level thickness, converting them into a `Sketch` composed of `Path`s and cubic Bézier `Curve`s. The canvas dimensions are dynamically calculated based on the bounding box of the strokes to ensure no clipping occurs.

**Source**: `datasets/linedrawer.py`

## Data Format

Each sample is a JSON array of strokes, where each stroke contains a sequence of points with the following attributes:

* `x`: x-coordinate (float)
* `y`: y-coordinate (float)
* `w`: local stroke thickness/width at the current point (float)
* `t`: time stamp

The loader groups these points into strokes and converts each consecutive point pair into a cubic Bézier curve, applying the `w` attribute to the `thickness` of the corresponding `Vertex`.

## Directory Layout

After download and extraction, the dataset is expected under:

```
<root>/
  LineDrawer/
    json/
      00000.json
      00001.json
      ...

```

* Each `.json` file in the `json/` directory corresponds to a single sketch.

## Code

```python
from sketchkit.datasets import LineDrawer

ds = SmartTracer()

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)

```