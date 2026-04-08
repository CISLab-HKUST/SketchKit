# SketchXPRIS

`SketchXPRIS` is a `SketchDataset` loader for the [SketchX-PRIS-Dataset](https://github.com/KeLi-SketchX/SketchX-PRIS-Dataset), which contains 20,000 drawings across 25 categories. It was collected by SketchX and PRIS to study universal perceptual grouping. Each drawing is stored as a sequence of strokes in stroke-3 format.

**Source**: `datasets/sketchx_pris.py`

## Data Format

Each category's data is stored in a `.ndjson` file. The sketches are parsed from stroke-3 format arrays `(N, 3)` containing:

- `dx`: delta x-coordinate, relative displacement from the previous point
- `dy`: delta y-coordinate, relative displacement from the previous point
- `pen_state`: pen state (0 for drawing / pen down, 1 for lifting / pen up)

The loader converts the relative coordinates into absolute coordinates, calculates the canvas size based on the bounding box (adding a 10% margin and ensuring a minimum 100x100 size), and groups continuous drawing segments into paths. Each stroke is represented as a cubic Bézier `Curve` within a `Path`.

## Directory Layout

After download and extraction, the dataset is expected under:

```text
<root>/
  SketchX-PRIS-Dataset-master/
    Group ID/
    Perceptual Grouping/
      apple.ndjson
      ice_cream.ndjson
      ...
  SketchX-PRIS.zip
  .metadata.parquet
```

- All sketch files are stored as `.ndjson` files categorized by class in the `Perceptual Grouping/` directory.
- The original archive `SketchX-PRIS.zip` is kept in the root directory for integrity checking.

## Code

```python
from sketchkit.datasets import SketchXPRIS

ds = SketchXPRIS(
    root="path/to/cache_dir",
    load_all=True,
    cislab_source=True,
)

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)

# Filter by category
ice_cream_metadata = ds.items_metadata[
    (ds.items_metadata["category"] == "ice_cream")
]
sketches = [ds[row.id] for _, row in ice_cream_metadata[:5].iterrows()]
```

**Arguments**

- `root`: Root directory used for caching and extraction.
- `load_all`: If `True`, preload all `.ndjson` arrays into memory to avoid disk I/O during iteration.
- `cislab_source`: Selects the download source. If `True`, the dataset is downloaded from the CISLAB CDN mirror (`SketchX-PRIS.zip`); otherwise it is downloaded directly from the official GitHub repository.

**Metadata Columns**

- `id`: Global unique identifier across all sketches.
- `sub_id`: Identifier within the specific category.
- `category`: Drawing category name (e.g., "ice_cream", "apple").

