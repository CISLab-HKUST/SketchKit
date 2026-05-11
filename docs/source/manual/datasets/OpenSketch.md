# OpenSketch

`OpenSketch` is a `SketchDataset` loader for the [OpenSketch](https://repo-sam.inria.fr/d3/OpenSketch) dataset, which contains more than 400 product design sketches representing 12 man-made objects drawn by 7 to 15 product designers of varying expertise. The dataset is designed for sketch-based modeling, nonphotorealistic rendering, and sketch filtering tasks. Each sketch is stored as an SVG file containing vector paths.

**Source**: `datasets/opensketch.py`

## Data Format

Each sample is an SVG file containing:

- `<path>` elements with `d` attributes defining drawing paths
- `viewBox` attribute defining canvas dimensions

The loader parses SVG paths and converts them into cubic Bézier curves. Supported SVG path commands include:
- `Move` (M): Starting point
- `Cubicezier` (C): Cubic Bézier curves
- `Line` (L): Straight lines (converted to cubic Bézier curves)

Each path is converted into a `Path` composed of cubic Bézier `Curve`s. The canvas dimensions are extracted from the SVG `viewBox` attribute.

## Directory Layout

After download and extraction, the dataset is expected under:

```
<root>/
  open-sketch_svgs/
    *.svg
  sketches_svg.zip
  .metadata.parquet
```

- All sketch files are stored as `.svg` files in the `opensketch_svgs/` directory.
- The original archive `sketches_svg.zip` is kept in the root directory for integrity checking.

## Code

```python
from sketchkit.datasets import OpenSketch

ds = OpenSketch(cislab_source=True)

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)
```

**Arguments**
- `cislab_source`: Selects the download source. If `True`, the dataset is downloaded from the CISLAB CDN mirror; otherwise it is downloaded from official host.
  - CISLAB mirror: `https://cislab.hkust-gz.edu.cn/projects/sketchkit/datasets/OpenSketch/sketches_svg.zip`
  - Official host: `https://repo-sam.inria.fr/d3/OpenSketch/Data/sketches/sketches_svgs.zip`