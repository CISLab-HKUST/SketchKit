# ControlSketch

`ControlSketch` is a `SketchDataset` loader for the [ControlSketch Dataset](https://swiftsketch.github.io/#controlsketch-data), introduced in the SwiftSketch paper (SIGGRAPH 2025).

**Source**: `datasets/controlsketch.py`

## Data Format

The SketchKit loader downloads the original dataset zip file, extracts the cached SVG content, parses each SVG file into a list of SketchKit `Path` objects, and then wraps those paths into a `Sketch` instance. The SVG parsing step reads `<path>` elements using:

- `M` (Move)
- `C` (CubicBezier)
- `L` (Line, converted to an equivalent cubic Bézier segment)

## Directory Layout

```
<root>/
  controlsketch_sketches/
    train/
      <category>/
        *.svg
    validation/
      <category>/
        *.svg
    test/
      <category>/
        *.svg
```

Splits are organized as `train`, `validation`, and `test`.

## Code

```python
from sketchkit.datasets import ControlSketch

ds = ControlSketch()

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)

bears = ds.items_metadata[
    (ds.items_metadata["category"] == "bear")
    & (ds.items_metadata["split"] == "validation")
]
bears_sketch = [ds[row.id] for _, row in bears[:100].iterrows()]
```

> **Note:** `ControlSketch()` automatically downloads the dataset from the [official host](https://drive.google.com/uc?id=1-kffChm6RXwTuP8qxUOT-QDf8-bSCH0w) if it is not already available locally.