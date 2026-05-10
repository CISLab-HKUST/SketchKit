# FSCOCO

`FSCOCO` is a `SketchDataset` loader for the [FS-COCO](https://fscoco.github.io/) dataset, where vector sketches are stored as `.npy` arrays. Each array encodes a sketch as polyline points with pen-state flags and is converted into a `Sketch` composed of `Path`s and cubic Bézier `Curve`s.

**Source**: `datasets/fscoco.py`

## Data Format

Each sample is a NumPy array with shape `(N, 3)`:

- `x`: x-coordinate (float)
- `y`: y-coordinate (float)
- `flag`: pen state (int)

Flag convention used by this loader:

- `0`: pen-down / continue current stroke
- `1`: pen-up (end current stroke; continue parsing)
- `2`: end-of-drawing (end current stroke; stop parsing)

The loader groups points into strokes and converts each consecutive point pair into a cubic Bézier curve.

## Directory Layout

After download and extraction, the dataset is expected under:

```
<root>/
  fscoco/
    vector_sketches/
      1/
        *.npy
      2/
        *.npy
      ...
      100/
        *.npy
  .metadata.parquet
```

- Shards are expected to be numbered **1..100**.
- Each shard contains multiple `.npy` sketch files.

## Code

```python
from sketchkit.datasets import FSCOCO

ds = FSCOCO()

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)
```