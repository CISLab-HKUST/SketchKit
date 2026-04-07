# PhotoSketching

`PhotoSketching` is a `SketchDataset` loader for the PhotoSketching dataset, where vector sketches are stored as `.svg` files. The dataset is designed for photo-to-sketch generation tasks, containing paired images and their corresponding sketch representations. It includes 1,000 outdoor photos paired with 5,000 SVG format sketches (5 sketches per photo). The dataset is particularly useful for supervised learning tasks that require paired photo-sketch data, such as sketch generation, style transfer, and sketch-based image retrieval. Each SVG sketch preserves geometric and stylistic information including cubic Bézier curves, stroke-level color attributes (RGB format), and stroke-level thickness attributes. The loader converts SVG files into a `Sketch` composed of `Path`s and cubic Bézier `Curve`s.

**Source**: `datasets/photosketching.py`

## Data Format

Each SVG file contains `<path>` elements with the following commands:
- `M` (Move): Starts a new path
- `C` (CubicBezier): Cubic Bézier curves with 4 control points
- `L` (Line): Line segments converted to cubic Bézier curves using 2/3-1/3 interpolation

Each path element includes:
- `d`: Path data containing Move and Bézier curve commands
- `stroke`: Stroke color in hexadecimal format (e.g., `#000` for black, converted to normalized RGB tuples)
- `stroke-width`: Stroke thickness/width (float, e.g., `3`)
- `fill`: Set to `none` (paths are stroked only)
- `id`: Path identifier (preserved but not used in SketchKit)
- `t`: Timestamp information (preserved but not used in SketchKit)

The loader extracts paths and converts them into SketchKit's `Sketch` format, preserving path-level color and thickness attributes. Canvas dimensions are extracted from SVG `viewBox` or `width`/`height` attributes.

**Limitations**:
- Transform attributes are not supported
- Arc segments are not supported
- Other SVG shapes (rect, circle, ellipse, line, polyline, polygon) are ignored
- Group (`<g>`) and other structural elements are parsed but only path data is extracted

## Directory Layout

After download and extraction, the dataset is expected under:

```
<root>/
  PhotoSketching/
    sketch/
      *.svg          # SVG format sketches
    sketch-rendered/
      *.png          # Rendered PNG sketches (3 line widths per SVG)
    .metadata.pkl    # Cached metadata file
```

* Each `.svg` file in the `sketch/` directory corresponds to a single sketch.
* `.metadata.pkl` is generated to index sketch file paths and metadata, containing columns: `id`, `svg_path`, `png_path`.

## Code

```python
from sketchkit.datasets import PhotoSketching

# Simple usage with default settings
ds = PhotoSketching()

# Or with custom parameters
# ds = PhotoSketching(
#     root="path/to/cache_dir",
#     load_all=False,
#     cislab_source=True,
# )

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)
```

**Arguments**

* `root`: Root directory used for caching and extraction. PhotoSketching data is placed under `<root>/PhotoSketching/`.
* `load_all`: If `True`, preload all `.svg` files into memory to avoid disk I/O and SVG parsing overhead during iteration.
* `cislab_source`: Selects the download source. If `True`, the dataset is downloaded from the CISLAB CDN mirror; otherwise it is downloaded from Google Drive.
  * CISLAB mirror: `https://cislab.hkust-gz.edu.cn/projects/sketchkit/datasets/PhotoSketching/all-in-one.zip`
  * Google Drive: `https://drive.google.com/file/d/1_AIxKnZXQms5Ezb-cEeVIDIoVG-eliHc`
