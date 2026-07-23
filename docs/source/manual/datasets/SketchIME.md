# SketchIME

`SketchIME` is a `SketchDataset` loader for the [SketchIME Dataset](https://dl.acm.org/doi/abs/10.1145/3581783.3612115), which provides over 56K drawings across 374 categories. 
The dataset provides semantic annotations across 139 parts.

**Source**: `datasets/sketchime.py`

## Data Format

Each sketch data is stored in a `.json` file. 
After loading the json file, we obtain a `dict` with the following keys:

- `key_id`: file name of the json.
- `recog_label`: category label, ranging from 0 to 373.
- `seg_label`: a list of semantic part labels ranging from 0 to 139, each belongs to a path. Note that 139 is an invalid flag for padding.
- `stroke_num`: a list of stroke numbers, each belongs to a path. Note that 0 is an invalid flag for padding.
- `sketch_stroke_num`: the number of valid paths.
- `points_offsets`: a list of point offsets on each path. Each path is a list of `[dx, dy, p1, p2]`. 
  
  - `[dx, dy]`: delta x/y coordinate, relative displacement from the previous point. The first one is [0.0, 0.0], which should be discarded.
  - `[p1, p2]`: can be [1.0, 0.0] (drawing / pen down), [0.0, 1.0] (lifting / pen up), and [0.0, 0.0] (for padding).

The loader converts the relative coordinates into absolute coordinates and then normalize them to a pre-defined image size.


## Directory Layout

After download and extraction, the dataset is expected under:

```text
<root>/
  .metadata.parquet
  SketchIME/
    cate_dir/
      001/
        C00101P08S001.json
        C00101P08S002.json
        ...
      002/
      ...
      374/
```

- All sketch files are stored as `.json` files in each category directory `001` to `374`.

## Code

```python
from sketchkit.datasets import SketchIME

ds = SketchIME(cislab_source=True)

sketch = ds[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)

# Filter by category
sketch_items = ds.items_metadata[
    (ds.items_metadata["category"] == "001")
]
sketches = [ds[row.id] for _, row in sketch_items[:5].iterrows()]
```

**Arguments**
- `cislab_source`: Selects the download source. If `True`, the dataset is downloaded from the CISLAB CDN mirror; otherwise it is downloaded from official host.
  - CISLAB mirror: `https://cislab.hkust-gz.edu.cn/projects/sketchkit/datasets/SketchIME/SketchIME.zip`
  - Official host: `https://drive.google.com/file/d/1TbaK46IQvI6MCs0JpCcB1Fcx8Nr3i8g8`


**Metadata Columns**

- `id`: Global unique identifier across all sketches.
- `sub_id`: Identifier within the specific category.
- `category`: Drawing category name (e.g., "001", "002", etc.).
- `filename`: The name of the json file.
- `split`: Split in the dataset (e.g., "train", "valid", or "test").
- `label_names`: A string of part label names, separated by ';' (e.g., "part1;part2;part3").