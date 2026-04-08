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

ds = Sketchy(
    root="path/to/cache_dir",
    load_all=True,
    cislab_source=True,
)

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)

# Filter out pruned (invalid) sketches for a specific class
valid_airplanes = ds.items_metadata[
    (ds.items_metadata["class_name"] == "airplane") & 
    (ds.items_metadata["pruned"] == False)
]
sketches = [ds[row.id] for _, row in valid_airplanes[:5].iterrows()]
```

**Arguments**

- `root`: Root directory used for caching and extraction. Data is extracted into `<root>/sketches/`.
- `load_all`: If `True`, preload all SVG paths into memory to avoid disk parsing during iteration.
- `cislab_source`: Selects the download source. If `True`, the dataset is downloaded from the CISLAB CDN mirror; otherwise it is downloaded from Google Drive.
  - CISLAB mirror: `https://cislab.hkust-gz.edu.cn/projects/sketchkit/datasets/Sketchy/sketches-06-04.7z`
  - Google Drive: File ID `1Qr8HhjRuGqgDONHigGszyHG_awCstivo`

**Metadata Columns**

- `id`: Global unique identifier across all sketches.
- `pruned`: Boolean flag indicating if the sketch was listed in the category's `invalid.txt` file.
- `class_name`: The class category of the sketch (e.g., "airplane").
- `filename`: The specific SVG file name (e.g., "n02691156_58-1.svg").
- `path`: The absolute path to the SVG file.
- `sketch_name`: The relative identifier combining class and filename (e.g., "airplane/n02691156_58-1.svg").