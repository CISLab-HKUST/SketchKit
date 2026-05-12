# DifferSketching

`DifferSketching` is a `SketchDataset` loader for the [DifferSketching](https://chufengxiao.github.io/DifferSketching/#dataset) dataset, which contains 3,620 freehand multi-view sketches from 70 novice users and 38 expert users sketching 136 3D objects. The dataset is designed to understand how differently people with and without adequate drawing skills sketch 3D objects. Each sketch is stored as a JSON file containing stroke sequences with pressure and timing information.

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

ds = DifferSketching()

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

> **Note:** `DifferSketching()` automatically downloads the dataset from the [official host](https://drive.google.com/file/d/1A_3RVc8Y4YdI7nhyM7tb-q7dQw4zTcCO) if it is not already available locally.