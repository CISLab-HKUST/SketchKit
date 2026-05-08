# Sketchy

`Sketchy` is a `SketchDataset` loader for the SVG subset of [The Sketchy Database](https://sketchy.eye.gatech.edu/). The dataset contains vector-based sketches categorized by class name. It natively handles pruning invalid or ambiguous sketches based on the dataset's provided annotation text files.

**Source**: `datasets/sketchy.py`

## Data Format

Each sample is an SVG file containing vector paths. The loader parses the XML tree and handles:

- `<path>` elements: The `d` attribute is parsed to extract control points. Supported commands include `Move` and `CubicBezier`. `Line` segments are automatically converted into cubic Bézier representations. 
- Dimensions: Canvas dimensions are extracted from the SVG `viewBox` or fallback `width`/`height` attributes.
- Unsupported elements: Primitive SVG shapes (like `rect`, `circle`, `ellipse`, `line`, `polyline`, `polygon`) and path transforms are ignored or unsupported by this parser.

Each path is converted into a `Path` composed of cubic Bézier `Curve`s to remain consistent with SketchKit's unified format.

## Directory Layout

After download and extraction, the dataset is expected under:

```text
<root>/
  sketches/
    airplane/
      n02691156_58-1.svg
      ...
      invalid.txt
    apple/
      ...
  sketches-06-04.7z
  .metadata.parquet
```

- `<root>/sketches/` contains subfolders for each class name.
- Each class folder contains its respective `.svg` files and an `invalid.txt` file listing unidentifiable or erroneous sketches (which are tracked via the `pruned` metadata column).

## Code

```python
from sketchkit.datasets import Sketchy

ds = Sketchy()

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)

# Filter out pruned (invalid) sketches for a specific class
valid_airplanes = ds.items_metadata[
    (ds.items_metadata["class_name"] == "airplane")
    & (ds.items_metadata["pruned"] == False)
]
sketches = [ds[row.id] for _, row in valid_airplanes[:5].iterrows()]
```

**Metadata Columns**

- `id`: Global unique identifier across all sketches.
- `pruned`: Boolean flag indicating if the sketch was listed in the category's `invalid.txt` file.
- `class_name`: The class category of the sketch (e.g., "airplane").
- `filename`: The specific SVG file name (e.g., "n02691156_58-1.svg").
- `path`: The absolute path to the SVG file.
- `sketch_name`: The relative identifier combining class and filename (e.g., "airplane/n02691156_58-1.svg").