# CreativeSketch

`CreativeSketch` is a `SketchDataset` loader for the [Creative Sketch dataset](https://songweige.github.io/projects/creative_sketech_generation/home.html), which contains vector sketches of birds and creatures with detailed part annotations.

**Source**: `datasets/creative_sketch.py`

## Data Format

The Creative Sketch dataset contains vector sketches with detailed part annotations. The SketchKit loader downloads the original dataset zip file, extracts the content, and parses each sketch from JSON files into a list of SketchKit `Path` objects, then wraps those paths into a `Sketch` instance.

Each sketch is represented as polylines, which are converted to cubic Bézier segments for consistency with other SketchKit datasets.

## Directory Layout

```
<root>/
  creative_sketch.zip
  raw_data_clean/
    creative_birds_json.txt
    creative_creatures_json.txt
  .metadata.parquet
```

- `creative_sketch.zip`: The original dataset zip file.
- `raw_data_clean/`: Directory containing JSON files with sketch data.
- `.metadata.parquet`: Cached metadata for all items in the dataset.

## Code

```python
from sketchkit.datasets import CreativeSketch

ds = CreativeSketch(
    cislab_source=True,
)

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)
```

**Arguments**

- `root`: Root directory used for caching and extraction. CreativeSketch data is placed under `<root>/`.
- `load_all`: If `True`, preload all parsed sketches into memory to reduce repeated disk I/O and parsing overhead.
- `cislab_source`: Selects the download source. If `True`, the dataset is downloaded from the CISLAB CDN mirror; otherwise it is downloaded from Google Drive.
  - CISLAB mirror: `https://cislab.hkust-gz.edu.cn/projects/sketchkit/datasets/CreativeSketch/creative_sketch.zip`
  - Official host: Google Drive (ID: 1Dstn3Cv2yAnxt1JXC0zPa1VPVDSz9gc-)

**Metadata Columns**

- `id`: Global unique identifier across all sketches.
- `sub_id`: Identifier within the specific category.
- `category`: Drawing category name (e.g., "bird", "creature").
- `is_good`: whether it is a good example drawn by a reliable worker.
- `split`: Split in the dataset (e.g., "train", "valid", or "test").
- `label_names`: A string of part label names, separated by ';' (e.g., "part1;part2;part3").



## Dataset Summary

The Creative Sketch dataset includes:

- **Bird sketches**: Vector sketches of birds with part annotations
- **Creature sketches**: Vector sketches of creatures with part annotations
- **Detailed part annotations**: Each stroke is labeled with its corresponding body part
- **Quality labels**: Each sketch is labeled as "good" or "not good"

## Usage Examples

### Basic Usage

```python
# Default Load
dataset = CreativeSketch()

print(f"Dataset summary:\n{dataset.extra_repr()}")
```

### Filtering Sketches

```python
# Select bird sketches
bird_indices = dataset.items_metadata[
    dataset.items_metadata["category"] == "bird"
].index
print(f"Number of bird sketches: {len(bird_indices)}")

# Select creature sketches
creature_indices = dataset.items_metadata[
    dataset.items_metadata["category"] == "creature"
].index
print(f"Number of creature sketches: {len(creature_indices)}")

# Select good samples
good_indices = dataset.items_metadata[
    dataset.items_metadata["is_good"] == True
].index
print(f"Number of good sketches: {len(good_indices)}")

# Combined filter: good bird samples
good_bird_indices = dataset.items_metadata[
    (dataset.items_metadata["category"] == "bird")
    & (dataset.items_metadata["is_good"] == True)
].index
print(f"Number of good bird sketches: {len(good_bird_indices)}")
```

### Loading and Rendering

```python
# Load a good bird sample
if len(good_bird_indices) > 0:
    bird_sketch = dataset[good_bird_indices[0]]
    print(f"Loaded bird sketch with {len(bird_sketch.paths)} paths")

# Load a creature sample
if len(creature_indices) > 0:
    creature_sketch = dataset[creature_indices[0]]
    print(f"Loaded creature sketch with {len(creature_sketch.paths)} paths")

# Render example
from PIL import Image
from sketchkit.renderer.cairo_renderer import CairoRenderer

renderer = CairoRenderer(800, (1, 1, 1))
if "bird_sketch" in locals():
    raster_image = renderer.render(bird_sketch)
    outpath = "test.png"
    if outpath is not None:
        raster_image_png = Image.fromarray(raster_image, "RGB")
        raster_image_png.save(outpath, "PNG")
        print(f"Sketch saved to {outpath}")
```
