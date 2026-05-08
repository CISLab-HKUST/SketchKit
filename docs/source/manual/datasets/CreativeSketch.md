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
from sketchkit.dss import CreativeSketch

ds = CreativeSketch()

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)

# Select bird sketches
bird_indices = ds.items_metadata[ds.items_metadata["category"] == "bird"].index
print(f"Number of bird sketches: {len(bird_indices)}")

# Select creature sketches
creature_indices = ds.items_metadata[ds.items_metadata["category"] == "creature"].index
print(f"Number of creature sketches: {len(creature_indices)}")

# Select good samples
good_indices = ds.items_metadata[ds.items_metadata["is_good"] == True].index
print(f"Number of good sketches: {len(good_indices)}")

# Combined filter: good bird samples
good_bird_indices = ds.items_metadata[
    (ds.items_metadata["category"] == "bird") & (ds.items_metadata["is_good"] == True)
].index
print(f"Number of good bird sketches: {len(good_bird_indices)}")
```

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