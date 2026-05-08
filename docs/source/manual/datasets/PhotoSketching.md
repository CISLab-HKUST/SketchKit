# PhotoSketching

`PhotoSketching` is a `SketchDataset` loader for the [Photo-Sketching: Inferring Contour Drawings from Images](https://mtli.github.io/sketch/) dataset (WACV 2019). The original dataset contains 1,000 outdoor images paired with 5,000 human contour drawings, and this loader parses the SVG sketches into a `Sketch` composed of `Path`s and cubic Bézier `Curve`s.

**Source**: `datasets/photosketching.py`

## Data Format

Each sample is stored as an SVG sketch. The loader reads canvas size from the SVG `viewBox` or `width` and `height` attributes, and parses `<path>` elements with the following commands:

* `M` (Move)
* `C` (CubicBezier)
* `L` (Line, converted to an equivalent cubic Bézier segment)

The path geometry is read from the SVG `d` attribute. The loader preserves path-level stroke color and thickness, and converts each parsed path into a sequence of cubic Bézier curves in SketchKit.

## Directory Layout

After download and extraction, the dataset is expected under:

```
<root>/
  PhotoSketching/
    sketch/
      *.svg
    sketch-rendered/
      width-1/
        *.png
      width-3/
        *.png
      width-5/
        *.png
    list/
      train.txt
      test.txt
      val.txt
    .metadata.pkl
```

Each `.svg` file under `sketch/` corresponds to a single vector sketch. The loader scans this directory recursively and caches indexed file paths in `.metadata.pkl`.

## Code

```python
from sketchkit.datasets import PhotoSketching

ds = PhotoSketching(
    cislab_source=True,
)

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)
```

**Arguments**

- `root`: Root directory used for caching and extraction. PhotoSketching data is placed under `<root>/`.
- `load_all`: If `True`, preload all parsed SVG sketches into memory to avoid repeated disk I/O and SVG parsing during iteration.
- `cislab_source`: Selects the download source. If `True`, the dataset is downloaded from the CISLAB CDN mirror; otherwise it is downloaded from Google Drive.
  - CISLAB mirror: `https://cislab.hkust-gz.edu.cn/projects/sketchkit/datasets/PhotoSketching/all-in-one.zip`
  - Google Drive: `https://drive.google.com/file/d/1_AIxKnZXQms5Ezb-cEeVIDIoVG-eliHc`
