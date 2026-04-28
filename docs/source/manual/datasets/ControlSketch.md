# ControlSketch

`ControlSketch` is a `SketchDataset` loader for the [ControlSketch Dataset](https://swiftsketch.github.io/#controlsketch-data), introduced in the SwiftSketch paper (SIGGRAPH 2025). 

**Source**: `datasets/controlsketch.py`

## Data Format

The SketchKit loader downloads the original dataset zip file, extracts the cached SVG content, parses each SVG file into a list of SketchKit `Path` objects, and then wraps those paths into a `Sketch` instance. The SVG parsing step reads `<path>` elements using:

* `M` (Move)
* `C` (CubicBezier)
* `L` (Line, converted to an equivalent cubic Bézier segment)

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

ds = ControlSketch(
    cislab_source=True,
)

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)
```

**Arguments**

- `root`: Root directory used for caching and extraction. ControlSketch data is placed under `<root>/controlsketch_sketches/`.
- `load_all`: If `True`, preload all parsed SVG sketches into memory to reduce repeated disk I/O and parsing overhead.
- `cislab_source`: Selects the download source. If `True`, the dataset is downloaded from the CISLAB CDN mirror; otherwise it is downloaded from Google Drive.
  - CISLAB mirror: `https://cislab.hkust-gz.edu.cn/projects/sketchkit/datasets/ControlSketch/controlsketch_sketches.zip`
  - Official host: `https://drive.google.com/drive/folders/1L5kubR416QoTD_UAqH2FtSgNL4leUcys`
