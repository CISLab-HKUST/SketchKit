# TracingVsFreehand

`TracingVsFreehand` is a `SketchDataset` loader for the [Tracing Versus Freehand](https://github.com/zachzeyuwang/tracing-vs-freehand) dataset (SIGGRAPH 2021), where vector sketches (including both tracings and freehand drawings) are stored as `.json` files. The raw data encodes sketches with continuous coordinate points and rich visual attributes, which are converted into a `Sketch` composed of `Path`s and cubic Bézier `Curve`s mapped to a fixed 800x800 canvas.

**Source**: `datasets/tracing_vs_freehand.py`

## Data Format

The dataset uses a nested JSON structure grouping strokes by prompt and participant. The loader extracts the following attributes for each point:

* `x`: x-coordinate (float)
* `y`: y-coordinate (float)
* `pressure`: normalized pen pressure (float)
* `width`: overall stroke thickness (float)
* `color`: hex color code converted to RGB tuple (float)
* `opacity`: alpha channel value (float)

The loader groups the parsed points into strokes and converts each consecutive point pair into a cubic Bézier curve, interpolating control points to form linear segments.

## Directory Layout

After download and extraction, the dataset is expected under:

```
<root>/
  TracingVsFreehand/
    .metadata_json.parquet
    sketch.zip
    data/
      drawings.json
      drawings_registered.json
      tracings.json

```

* The dataset is divided into three primary subset JSON files.
* `.metadata_json.parquet` is generated to index the hierarchical JSON structure.

## Code

```python
from sketchkit.datasets import TracingVsFreehand

ds = TracingVsFreehand(
    cislab_source=True,
)

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)

```

**Arguments**

- `root`: Root directory used for caching and extraction. TracingVsFreehand data is placed under `<root>/TracingVsFreehand/`.
- `load_all`: If `True`, preload all parsed sketches from the JSON files into memory to avoid JSON decoding overhead during iteration.
- `cislab_source`: Selects the download source. If `True`, the dataset is downloaded from the CISLAB CDN mirror.
  - CISLAB mirror: `https://cislab.hkust-gz.edu.cn/projects/sketchkit/datasets/TracingVsFreehand/sketch.zip`