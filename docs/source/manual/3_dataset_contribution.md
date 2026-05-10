# Dataset Contribution Guide

This guide provides a comprehensive walkthrough for developers who want to contribute a new or customized dataset to the SketchKit toolkit.

## Phase 1: Collaboration Workflow (GitHub Flow)

We utilize the **GitHub Flow** to manage contributions. Please ensure your local environment is configured with Git and your GitHub credentials.

### 1. Fork the Repository

If you don't have write access to the main SketchKit repository, fork it to your personal account.

- Click the **Fork** button on the GitHub repository page.
- Clone your fork to your local machine:

```
git clone https://github.com/{your_username}/SketchKit.git
cd SketchKit
```

### 2. Create a Branch

Create a branch with a descriptive name related to the task or method you are adding.

```
git checkout -b new-dataset
```

### 3. Make Changes

See Phase 2 below for instructions on writing the code. Then commit the changes.

```
git add -A .
git commit -m "add new dataset: {dataset_name}."
```

### 4. Push Changes

Upload your branch to your remote fork.

```
git push -u origin new-dataset
```

### 5. Create a Pull Request (PR)

On GitHub, open a **Pull Request**.

- Go to your branch and click New pull request.
- Summarize changes, link related issues, and add visuals if helpful.
- Use draft PR for early feedback.

### 6. Address Review Comments

If reviewers suggest changes, apply them locally and push again. The PR updates automatically.

```
git add -A .
git commit -m "updated dataset: {dataset_name}."
git push -u origin new-dataset
```

### 7. Merge & Cleanup

Once merged, sync your local environment and delete the feature branch.

```
git switch main
git pull
git branch -d new-dataset
```

## Phase 2: Technical Integration Workflow

Prepare the data and code scripts according to the following steps.

### 1. Prepare Your Data Package

a) Prepare the vector data of the customized dataset in polyline or Bezier curve representation.

b) Archive the data package, for example, in ZIP format.

c) Compute the MD5 sum of the whole directory.

d) Obtain the download link of the package.

### 2. Create The Dataset Script

a) Go to `sketchkit/datasets` directory, and create a `{my_dataset}.py` file. Use your own name.

b) Define a `MyDataset` class (use your own name), which inherits from an abstract class `SketchDataset`. Add doc-strings, including:

- Description of the dataset, including its source, its composition, and the representation of the sketches
- Attributes: such as `md5_sum` and `metadata`
- References: the URL and paper title of the dataset

```python
class MyDataset(SketchDataset):
    """The MyDataset dataset ...

    Attributes:
        md5_sum (str): MD5 checksum for dataset integrity verification.
        metadata (list[str]): Column names used in the metadata table.

    References:
        - Original dataset: www.dataset-download.com
        - Paper: xxx
    """
```

c) Update the `__init__.py` in `sketchkit/datasets` directory. Add the following line. Then add `"MyDataset"` into `__all__`.

```python
from .my_dataset import MyDataset

__all__ = [
    ...,
    "MyDataset"
]
```

### 3. Implement Overridden Methods

The `MyDataset` is inherited from an abstract class `SketchDataset`. The overridden methods should be implemented following the steps below. You can refer to the implementations of {ref}`other datasets <dataset-feature>` for a reference.

a) Add Attributes of the `MyDataset` class, including:

- md5_sum: the computed MD5 sum in Step 1
- metadata ({ref}`reference <dataset-metadata>`): it varies across datasets. Common metadata includes:
  - "category": category of a sketch
  - "split": split in the dataset, such as "train", "val", or "test"
  - "id": Global unique identifier across all sketches
  - "sub_id": Identifier within the category and split

b) `_download()`: a method for downloading the data package automatically.

- Define the URL and path for caching (e.g., `self.root/mydata_package.zip`).
- Download with the method `download_with_wget` and extract with `extract_files`. They are provided in `sketchkit.utils.file`. Extraction is not needed for a data package stored in npz/npy format.

c) `_check_integrity() -> bool`: a method for checking the integrity of the cached dataset using MD5 checksum.

- Compute the MD5 with `file_md5` method provided in `sketchkit.utils.file`
- Determine whether the computed MD5 is the same as the pre-defined MD5 in the Attributes of the `MyDataset` class
- Return True or False

d) `_load_items_metadata()`: a method for loading and caching metadata for all items in the dataset.

- Creates a parquet file (`self.root/.metadata.parquet`) containing metadata defined above for all sketches
- If `.metadata.parquet` does not exist, use the code below:

```python
items_metadata = pd.DataFrame(columns=["category", "split", "id", "sub_id"])
new_rows = [
    {
        "category": {category_A},
        "split": {split_A},
        "id": {global_id},
        "sub_id": {file_id},
    },
    {
        "category": {category_A},
        "split": {split_A},
        "id": {global_id},
        "sub_id": {file_id},
    },
    ...
]
items_metadata = pd.concat(
    [items_metadata, pd.DataFrame(new_rows)], ignore_index=True
)
self.items_metadata = items_metadata
self.items_metadata.to_parquet(
    os.path.join(self.root, ".metadata.parquet"), compression="zstd"
)
```

- If `.metadata.parquet` exists, use the code below:

```python
items_metadata = pd.read_parquet(
    os.path.join(self.root, ".metadata.parquet")
)
self.items_metadata = items_metadata
```

e) `_load_all()`: a method for loading all sketch data into memory ({ref}`reference <key-concepts-of-dataset-parameters>`).

- Traverse all the sketches, obtain their widths and heights, and parse them into path lists (see Step 4 below for details)
- Fill each sketch data into `self.raw_data`

```python
(sketch_width, sketch_height), path_list = parse_sketch(sketch_path)
self.raw_data[cnt] = [path_list, (sketch_width, sketch_height)]
```

f) `_get_single(idx: int) -> Sketch`: a method for getting a sketch by index.

- Input: `idx (int)`: Index of the sketch to retrieve.
- Output: `Sketch`: A Sketch object containing the drawing data as paths.
- Load its path list and obtain its width and height. And then convert it into the `Sketch` instance (see details of the [representation](./1_representation.md)) by constructing `Vertex/Point`, `Curve`, `Path`, and `Sketch`

```python
path_list, (width, height) = self.raw_data[idx]
paths = []
for path in path_list:
    curves = []
    for raw_curve in path:
        p_start = Vertex(raw_curve[0][0], raw_curve[0][1])
        p_end = Vertex(raw_curve[3][0], raw_curve[3][1])
        p_ctrl1 = Point(raw_curve[1][0], raw_curve[1][1])
        p_ctrl2 = Point(raw_curve[2][0], raw_curve[2][1])
        curves.append(Curve(p_start, p_end, p_ctrl1, p_ctrl2))
    paths.append(Path(curves=curves))
sketch = Sketch(height=height, width=width, paths=paths)
```

g) `extra_repr() -> str`: a method for returning extra information in the string representation.

- Such as the number of categories, the number of sketches in each split.

### 4. Parsing Sketch Data

The [sketch representation](./1_representation.md) in SketchKit is organized in a hierarchy of `Sketch` - `Path` - `Curve` - `Vertex/Point`, so you need to parse the raw sketch data according to this hierarchy by constructing a list of paths. Each path is a list of curves (cubic Bezier curve). Each curve is a list of four points. Each point is (x, y) in absolute coordinates.

Note: If the raw sketch data is represented in polylines, they should be converted into cubic Bezier curves. Specifically, the two intermediate control points are defined as the 1/3 and 2/3 positions on the polyline. Refer to the <a href="../_modules/sketchkit/core/sketch.html#Curve.from_line">Curve.from_line</a> method for details.

## Phase 3: Testing

Use the `sketchkit/tests/test_dataset.py` for testing the integrated dataset.

Follow the steps below:

a) Initialize a dataset instance

```python
from sketchkit.datasets import MyDataset
dataset = MyDataset()
```

b) Load a sketch example using an index or from a specified category and a specified split

```python
# Load a sketch using index
sketch = dataset[0]

# Load a sketch from a specified category and a specified split
subset = dataset.items_metadata[
    (dataset.items_metadata["category"] == "cat") & (dataset.items_metadata["split"] == "train")
]
subset_sketch = [dataset[row.id] for _, row in subset[:100].iterrows()]
sketch = subset_sketch[0]
```

c) Output the attributes of the sketch, such as width, height, path_num, curve_num, etc.

```python
print(sketch.width, sketch.height)
print(sketch.path_num, sketch.curve_num)
```

d) Render the vector sketch into a raster image using the built-in renderer (refer to the [document](./14_renderer.md) for details)

```python
from sketchkit.renderer import CairoRenderer
renderer = CairoRenderer()
raster_image = renderer.render(sketch)
raster_image.save("sketch.png")
```

## Phase 4: Documentation

A manual document for the new dataset is required. Follow the steps below to write the document. You can refer to the documents of {ref}`Other documents <available-datasets>` for a reference.

a) Create a markdown file `MyDataset.md` in the `docs/source/manual/datasets` directory.

b) The document includes the following parts:

- **Introduction** to the dataset: its source (URL), composition, number of categories, number of examples, representation of the raw sketch data, etc.
- **Source** of the dataset script: `datasets/my_dataset.py`
- **Data Format**: introduce the original data format of the sketches.
- **Directory Layout**: the file structure after downloading the dataset.
- **Code**: code snippet of quick usage of the dataset, including initializing a dataset instance, loading a sketch example, outputting the attributes, and rendering the vector sketch (similar to the testing).
- **Arguments**: introduction of necessary arguments in the code.
