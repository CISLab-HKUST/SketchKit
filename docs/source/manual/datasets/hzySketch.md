# hzySketch

`hzySketch` is a `SketchDataset` loader for the hzy dataset, where the sequential drawing process of high-quality anime line arts is stored as `.json` files. The loader extracts coordinate points and dynamic point-level thickness, converting them into a `Sketch` composed of `Path`s and cubic Bézier `Curve`s. The canvas dimensions are dynamically calculated based on the bounding box of the strokes to ensure no clipping occurs.

**Source**: `datasets/hzy_sketch.py`

## Data Format

Each sample is a JSON array of strokes, where each stroke contains a sequence of points with the following attributes:

* `x`: x-coordinate (float)
* `y`: y-coordinate (float)
* `w`: local stroke thickness/width at the current point (float)

The loader groups these points into strokes and converts each consecutive point pair into a cubic Bézier curve, applying the `w` attribute to the `thickness` of the corresponding `Vertex`.

## Directory Layout

After download and extraction, the dataset is expected under:

```
<root>/
  hzySketch/
    .metadata.parquet
    hzy_sketch.zip
    json/
      00000.json
      00001.json
      ...

```

* Each `.json` file in the `json/` directory corresponds to a single sketch.

## Code

```python
from sketchkit.datasets import hzySketch

ds = hzySketch(
    root="path/to/cache_dir",
    load_all=False,
    cislab_source=True,
)

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)

```

**Arguments**

* `root`: Root directory used for caching and extraction. hzySketch data is placed under `<root>/hzySketch/`.
* `load_all`: If `True`, preload all `.json` files into memory to avoid disk I/O and JSON decoding overhead during iteration.
* `cislab_source`: Selects the download source. If `True`, the dataset is downloaded from the CISLAB CDN mirror.
* CISLAB mirror: `https://cislab.hkust-gz.edu.cn/projects/sketchkit/datasets/hzySketch/hzy_sketch.zip`