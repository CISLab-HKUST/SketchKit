# CreativeSketch

`CreativeSketch` is a `SketchDataset` loader for the [Creative Sketch](https://songweige.github.io/projects/creative_sketech_generation/home.html) dataset, which contains vector sketches of birds and creatures with detailed part annotations. Each sketch is stored as JSON data and is converted into a `Sketch` composed of `Path`s and linear `Curve`s with color attributes.

**Source**: `datasets/creative_sketch.py`

## Data Format

Each sample contains stroke sequence data with part annotations:

- `all_strokes`: List of stroke sequences forming the sketch
- `partsUsed`: List indicating which body part each stroke represents (e.g., "head", "wing", "body")
- `good_sample`: Boolean flag indicating if the sample quality is good

The dataset supports two categories:

- **Birds**: With part labels including "head", "wings", "beak", "eye", "body", "tail", "legs", "mouth", "details", etc.
- **Creatures**: With part labels including "head", "wings", "arms", "body", "ears", "feet", "hair", "hands", "horns", "legs", "nose", "paws", "tail", etc.

Each stroke is represented as a sequence of line segments, which are converted to cubic Bézier curves with color attributes based on the part being drawn.

## Color Mapping

The dataset assigns distinct colors to different body parts:

- **Bird colors**: Different colors for head (red), wings (gray), beak (orange), eye (yellow), body (blue), tail (dark gray), legs (purple), etc.
- **Creature colors**: Different colors for head (gold), wings (light gray), arms (orange), body (red), ears (purple), hair (gray), horns (green), etc.

## Directory Layout
- `creative_birds_json.txt`: Contains JSON data for bird sketches
- `creative_creatures_json.txt`: Contains JSON data for creature sketches
- Each JSON file contains an array of sketch objects with stroke and annotation data

## Metadata

The dataset provides the following metadata columns:

- `id`: Global unique identifier for each sketch
- `category`: Category of the sketch ("bird" or "creature")
- `sub_id`: Index within the category-specific JSON file
- `is_good`: Boolean indicating if the sketch is a good sample

## Code

```python
from sketchkit.datasets import CreativeSketch

ds = CreativeSketch(
    root="path/to/cache_dir",
    load_all=True,
    cislab_source=True,
)

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)

# Example: Filter by category
bird_indices = ds.items_metadata[
    ds.items_metadata["category"] == "bird"
].index
bird_sketch = ds[bird_indices[0]]

# Example: Filter good samples only
good_indices = ds.items_metadata[
    ds.items_metadata["is_good"] == True
].index
good_sketch = ds[good_indices[0]]

# Example: Get creature sketches
creature_indices = ds.items_metadata[
    ds.items_metadata["category"] == "creature"
].index
creature_sketch = ds[creature_indices[0]]
```

**Arguments**

- `root`: Root directory used for caching and extraction. CreativeSketch data is placed under `<root>/`.
- `load_all`: If `True`, preload all JSON data into memory to avoid disk I/O during iteration.
- `cislab_source`: Selects the download source. If `True`, the dataset is downloaded from the CISLAB CDN mirror; otherwise it is downloaded from the official host.
  - CISLAB mirror: `https://cislab.hkust-gz.edu.cn/projects/sketchkit/datasets/CreativeSketch/creative_sketch.zip`
  - Official host: Google Drive ID `1Dstn3Cv2yAnxt1JXC0zPa1VPVDSz9gc-`