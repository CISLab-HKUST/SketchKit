# QuickDraw

`QuickDraw` is a `SketchDataset` loader for the [Quick Draw](https://github.com/googlecreativelab/quickdraw-dataset) dataset. It contains millions of drawings across 345 categories, collected from the Quick, Draw! game.
The vector sketches of each category are stored in an `npz` file, including 75K samples (70K Training, 2.5K Validation, 2.5K Test). Each drawing is represented as a sequence of strokes in stroke-3 format.

**Source**: `datasets/quickdraw.py`

## Data Format

Each sketch sample is a NumPy array with shape `(N, 3)`:

- `dx`: delta x-coordinate (float), relative displacement from the previous coordinate
- `dy`: delta y-coordinate (float), relative displacement from the previous coordinate
- `flag`: pen state (int)

Flag convention used by this loader:

- `0`: pen-down (for drawing)
- `1`: pen-up (for lifting)

The loader converts the format into absolute coordinates and groups continuous drawing segments into paths, with each stroke represented as a cubic Bézier curve.

## Directory Layout

After downloading, the dataset is stored in the following structure:

```
<root>/
  QuickDraw/
    .metadata.parquet
    categories.txt
    aircraft carrier.npz
    airplane.npz
    ...
    zigzag.npz
```

## Code

```python
from sketchkit.datasets import QuickDraw

dataset = QuickDraw()

# search data with "category = cat" and "split = train"
cats = dataset.items_metadata[
    (dataset.items_metadata["category"] == "cat")
    & (dataset.items_metadata["split"] == "train")
]
# fetch the first 100 cat sketches
cats_sketch = [dataset[row.id] for _, row in cats[:100].iterrows()]
del dataset, cats

# Load a sketch using index
sketch = cats_sketch[0]
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)
```

> **Note:** `QuickDraw()` automatically downloads all categories from the [official host](https://storage.googleapis.com/quickdraw_dataset/sketchrnn/<category>.npz) if they are not already available locally, by iterating through `categories.txt` and fetching every `.npz` file; the initial download is therefore very large and may take a long time.

**Metadata Columns**

- `id`: Global unique identifier across all sketches.
- `sub_id`: Identifier within the specific category.
- `category`: specify the category name in the `categories.txt`, such as "cat", "airplane", "The Great Wall of China", etc.
- `split`: "train", "valid", or "test".