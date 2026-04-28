# DifferSketching

`DifferSketching` is a `SketchDataset` loader for the [DifferSketching](https://differsketching.github.io/) dataset, which contains 3,620 freehand multi-view sketches from 70 novice users and 38 expert users sketching 136 3D objects. The dataset is designed to understand how differently people with and without adequate drawing skills sketch 3D objects. Each sketch is stored as a JSON file containing stroke sequences with pressure and timing information.

**Source**: `datasets/differsketching.py`

## Data Format

Each sample is a JSON file containing:

- `strokes`: Array of stroke objects, where each stroke contains:
  - `path`: Array of `[x, y]` coordinate pairs (float)
  - `pressure`: Array of pressure values (float, optional)
  - `width`: Stroke thickness (float, optional)
- `width`: Canvas width (float, optional)
- `height`: Canvas height (float, optional)

The loader converts each stroke into a `Path` composed of cubic Bézier `Curve`s. Points are shifted so that the minimum x and y coordinates are at (0, 0).

## Directory Layout

After download and extraction, the dataset is expected under:

```
<root>/
  DifferSketching_Dataset/
    [Category]/
      original_json/
        *.json
      global_json/
        *.json
      stroke_json/
        *.json
      reg_json/
        *.json
  .metadata.parquet
```

- `[Category]` indicates the category name (e.g., `Animal_Head`).
- Drawing types:
  - `original`: Original user drawings
  - `global`: Sketch-level registered drawings
  - `stroke`: Stroke-level registered drawings
  - `reg`: Pixel-level registered drawings

## Code

```python
from sketchkit.datasets import DifferSketching

ds = DifferSketching(
    cislab_source=True,
)

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)

# Filter by category and type
objects = ds.items_metadata[
    (ds.items_metadata["category"] == "Animal_Head")
    & (ds.items_metadata["type"].isin(["original", "global"]))
]
sketches = [ds[row.id] for _, row in objects[:5].iterrows()]
```

**Arguments**

- `root`: Root directory used for caching and extraction. DifferSketching data is placed under `<root>/DifferSketching_Dataset/`.
- `load_all`: If `True`, preload all JSON files into memory to avoid disk I/O during iteration.
- `cislab_source`: Selects the download source. If `True`, the dataset is downloaded from the CISLAB CDN mirror; otherwise it is downloaded from Google Drive.
  - CISLAB mirror: `https://cislab.hkust-gz.edu.cn/projects/sketchkit/datasets/DifferSketching/differsketching.zip`
  - Google Drive: File ID `1A_3RVc8Y4YdI7nhyM7tb-q7dQw4zTcCO`
