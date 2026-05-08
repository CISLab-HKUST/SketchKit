# TUBerlin

`TUBerlin` is a `SketchDataset` loader for the [TU Berlin](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/) dataset, where vector sketches are stored in `.svg` files. 
The sketches are represented with cubic Bézier curves with four control points. 
The dataset includes sketches across 250 categories.
Each category contains 800 sketches. 

**Source**: `datasets/tu_berlin.py`

## Data Format

Each SVG file includes a series of `path`.
Each `path` comprises commands `M` (Move), `C` (CubicBezier), and `L` (Line).

## Directory Layout

After download and extraction, the dataset is expected under:

```
<root>/
  TUBerlin/
    .metadata.parquet
    sketches_svg.zip
    svg/
      airplane/
        *.svg
      backpack/
        *.svg
      ...
      zebra/
        *.svg
```

## Code

```python
from sketchkit.datasets import TUBerlin

ds = TUBerlin()

# Load a sketch using index
sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)

## Visualization
from sketchkit.renderer import CairoRenderer

# Initialize a renderer for rendering
renderer = CairoRenderer()
raster_image = renderer.render(sketch)
raster_image.save("sketch.png")
```

**Arguments**

- `root`: Root directory used for caching and extraction.
- `load_all`: If `True`, preload all data into memory to avoid disk I/O during iteration.

**Metadata Columns**

- `id`: Global unique identifier across all sketches.
- `sub_id`: Identifier within the specific category.
- `category`: Drawing category name (e.g., "airplane", "alarm clock", etc.).
- `split`: "train", "valid", or "test".