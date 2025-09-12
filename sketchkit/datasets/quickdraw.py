import os
import shutil

import numpy as np
from tqdm import tqdm
from pathlib import Path as PathLibPath
import pandas as pd


from sketchkit.core.sketch import Sketch, Path
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.transform import line_to_cubic
from sketchkit.utils.file import download_with_wget, dir_md5, CISLAB_CDN


def parse_stroke3_format(stroke_data):
    """Convert stroke-3 format in QuickDraw into SketchKit's Sketch format.

    The stroke-3 format consists of points with (dx, dy, pen_state) where:
    - dx, dy: relative displacement from the previous point
    - pen_state: 0 for drawing (pen down), 1 for lifting (pen up)

    This function converts the format into absolute coordinates and groups
    continuous drawing segments into paths, with each stroke represented
    as a cubic Bezier curve.

    Args:
        stroke_data (np.ndarray): Array of shape (N_point, 3), where each point
            is in stroke-3 format (dx, dy, pen_state). Pen state is 0 (drawing)
            or 1 (lifting).

    Returns:
        tuple: A tuple containing:
            - path_list (list[Path]): List of Path objects, each containing
              stroke segments as cubic Bezier curves.
            - total_segment_num (int): Total number of stroke segments across
              all paths.
    """
    dx = stroke_data[:, 0]  # (N_point)
    dy = stroke_data[:, 1]  # (N_point)
    pen_state = stroke_data[:, 2]  # (N_point)

    # Convert to absolute coordinates
    xs, ys = np.cumsum(dx), np.cumsum(dy)
    xs -= np.min(xs)
    ys -= np.min(ys)
    xy = np.stack([xs, ys], axis=1)  # (N_point, 2)

    path_list = []  # list of (N_strokes, 4, 2)
    stroke_list = []  # list of (4, 2)
    total_segment_num = 0
    for i in range(len(pen_state) - 1):
        if pen_state[i] == 0:
            p_start = xy[i]  # (2)
            p_end = xy[i + 1]  # (2)
            line = np.stack([p_start, p_end], axis=0)  # (2, 2)
            cubic = line_to_cubic(line)  # (4, 2)
            stroke_list.append(cubic)
            total_segment_num += 1
        else:
            # path_list.append(np.stack(stroke_list, axis=0))
            path_list.append(Path(stroke_list))
            stroke_list = []
    return path_list, total_segment_num


class QuickDraw(SketchDataset):
    """QuickDraw dataset loader and interface.

    The QuickDraw dataset contains millions of drawings across 345 categories,
    collected from the Quick, Draw! game. Each drawing is represented as a
    sequence of strokes in stroke-3 format.

    Attributes:
        md5_sum (str): MD5 checksum for dataset integrity verification.

    References:
        - Original dataset: https://github.com/googlecreativelab/quickdraw-dataset
        - Paper: "The Quick, Draw! Dataset" by Ha and Eck, 2017
    """

    # MD5 Sum of the whole dir, you can set a list for zip files
    md5_sum = "4d3c05094288833fccc0d28af45af4cd"
    # Metadata columns for each items
    metadata = ["id", "sub_id", "category", "split"]

    def __init__(
        self,
        root: str | PathLibPath | None = None,
        load_all: bool = False,
        cislab_source: bool = False,
    ):
        """Initialize the QuickDraw dataset.

        Args:
            root (str | PathLibPath | None): Root directory to store/load the dataset.
                If None, uses the default dataset directory.
            load_all (bool): If True, loads all data into memory for faster access.
                If False, loads data on-demand. Defaults to False.
            cislab_source (bool): If True, downloads from CISLAB CDN mirror.
                If False, downloads from original Google storage. Defaults to False.
        """
        super().__init__(root, load_all=load_all, cislab_source=cislab_source)

    def _check_integrity(self) -> bool:
        """Check the integrity of the cached dataset using MD5 checksum.

        Returns:
            bool: True if the dataset integrity is verified, False otherwise.
        """
        print(f"Checking integrity of cached {self.__class__.__name__} dataset...")
        current_md5 = dir_md5(self.root)
        return current_md5 == self.md5_sum

    def _download(self):
        """Download the QuickDraw dataset.

        Downloads the categories.txt file and all category .npz files from either
        the original Google storage or CISLAB CDN mirror. Creates the root directory
        if it doesn't exist.

        Raises:
            Exception: If download fails for any file.
        """

        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

        category_txt = os.path.join(self.root, "categories.txt")
        try:
            if self.cislab_source:
                url = (
                    f"{CISLAB_CDN}/datasets/{self.__class__.__name__}/"
                    + "categories.txt"
                )
            else:
                url = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/refs/heads/master/categories.txt"
            download_with_wget(
                url,
                file_path=category_txt,
                pg_bar=False,
            )
        except Exception as e:
            raise e

        all_categories = []
        with open(category_txt, "r") as f:
            for line in f.readlines():
                line = line.strip("\n")
                all_categories.append(line)

        for category in tqdm(all_categories, desc="Downloading categories"):
            npz_path = os.path.join(self.root, category + ".npz")
            try:
                if self.cislab_source:
                    url = (
                        f"{CISLAB_CDN}/datasets/{self.__class__.__name__}/"
                        + category
                        + ".npz"
                    )
                else:
                    url = (
                        "https://storage.googleapis.com/quickdraw_dataset/sketchrnn/"
                        + category
                        + ".npz"
                    )
                download_with_wget(url, npz_path, desc=f"Downloading {category}")

            except Exception as e:
                raise e
        print("MD5 Checksum:", dir_md5(self.root))

    def _load_items_metadata(self):
        """Load and cache metadata for all items in the dataset.

        Creates a parquet file containing metadata for all sketches including
        category, split (train/valid/test), global ID, and sub ID within category.
        If metadata cache exists, loads from the cached file.

        The metadata DataFrame contains columns:
        - category: Drawing category name
        - split: Data split (train/valid/test)
        - id: Global unique identifier across all sketches
        - sub_id: Identifier within the category and split
        """
        if not os.path.exists(os.path.join(self.root, ".metadata.parquet")):
            category_txt = os.path.join(self.root, "categories.txt")
            all_categories = []
            with open(category_txt, "r") as f:
                for line in f.readlines():
                    line = line.strip("\n")
                    all_categories.append(line)
            self._all_categories = sorted(all_categories)
            items_metadata = pd.DataFrame(columns=["category", "split", "id", "sub_id"])
            for category in tqdm(self._all_categories, desc="Caching metadata"):
                npz_path = os.path.join(self.root, category + ".npz")
                data = np.load(npz_path, encoding="latin1", allow_pickle=True)
                for split in ["train", "valid", "test"]:
                    l = len(items_metadata)
                    new_rows = [
                        {
                            "category": category,
                            "split": split,
                            "id": l + id,
                            "sub_id": id,
                        }
                        for id in range(data[split].shape[0])
                    ]
                    items_metadata = pd.concat(
                        [items_metadata, pd.DataFrame(new_rows)], ignore_index=True
                    )
            self.items_metadata = items_metadata
            self.items_metadata.to_parquet(
                os.path.join(self.root, ".metadata.parquet"), compression="zstd"
            )
        else:
            items_metadata = pd.read_parquet(
                os.path.join(self.root, ".metadata.parquet")
            )
            if not set(items_metadata.columns.tolist()) == set(self.metadata):
                os.remove(os.path.join(self.root, ".metadata.parquet"))
                return self._load_items_metadata()
            self.items_metadata = items_metadata
            self._all_categories = sorted(
                self.items_metadata["category"].unique().tolist()
            )
        self.raw_data = [None] * len(self.items_metadata)

    def _load_all(self):
        """Load all sketch data into memory if load_all is enabled.

        Concatenates all sketch data from all categories and splits into a single
        numpy array for faster access. Only loads if self.load_all is True.
        """
        cnt = 0
        for category in tqdm(self._all_categories, desc="Loading all data to memory"):
            npz_path = os.path.join(self.root, category + ".npz")
            data = np.load(npz_path, encoding="latin1", allow_pickle=True)
            for split in ["train", "valid", "test"]:
                self.raw_data[cnt : cnt + data[split].shape[0]] = data[split].tolist()
                cnt += data[split].shape[0]

    def __getitem__(self, idx: int) -> Sketch:
        """Get a sketch by index.

        If a sketch not in memory, load all sketches in the same category from disk.

        Args:
            idx (int): Index of the sketch to retrieve.

        Returns:
            Sketch: A Sketch object containing the drawing data as paths.

        Raises:
            IndexError: If index is out of range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # check if in memory, if not load from disk, along with all items in the same category to avoid I/O cost
        if self.raw_data[idx] is None:
            item_metadata = self.items_metadata.iloc[idx]
            npz_path = os.path.join(self.root, item_metadata["category"] + ".npz")
            data = np.load(npz_path, encoding="latin1", allow_pickle=True)
            # find the min id of item in the same category
            cnt = self.items_metadata[
                self.items_metadata["category"] == item_metadata["category"]
            ]["id"].min()
            for split in ["train", "valid", "test"]:
                self.raw_data[cnt : cnt + data[split].shape[0]] = data[split].tolist()
                cnt += data[split].shape[0]

        single_sketch_stroke = self.raw_data[idx]
        path_list, total_segment_num = parse_stroke3_format(single_sketch_stroke)
        sketch = Sketch(paths=path_list)
        return sketch

    def extra_repr(self) -> str:
        num_train = len(self.items_metadata[self.items_metadata["split"] == "train"])
        num_test = len(self.items_metadata[self.items_metadata["split"] == "test"])
        num_val = len(self.items_metadata[self.items_metadata["split"] == "valid"])
        return f"Categories: {len(self._all_categories)}\nTrain: {num_train} Validation: {num_val} Test samples: {num_test} "


if __name__ == "__main__":
    import psutil
    from rich.console import Console
    import time

    console = Console()

    console.print("QuickDraw Dataset Test", style="bold blue")
    console.print()

    # console.print("1. Default Load", style="green")
    # dataset = QuickDraw()
    # # show brief information of the dataset
    # console.print(dataset)
    # process = psutil.Process()
    # memory_mb = process.memory_info().rss / 1024 / 1024
    # console.print(f"Current memory usage: {memory_mb:.2f} MB")

    # # search data with "category = cat" and "split = train"
    # cats = dataset.items_metadata[
    #     (dataset.items_metadata["category"] == "cat")
    #     & (dataset.items_metadata["split"] == "train")
    # ]
    # start_time = time.time()
    # cats_sketch = [dataset[row.id] for _, row in cats[:100].iterrows()]
    # console.print(f"Loading 100 sketches in {time.time() - start_time}")
    # del dataset, cats

    # console.print()

    # console.print("2. Load All", style="green")
    # dataset = QuickDraw(load_all=True)
    # # show brief information of the dataset
    # console.print(dataset)
    # process = psutil.Process()
    # memory_mb = process.memory_info().rss / 1024 / 1024
    # console.print(f"Current memory usage: {memory_mb:.2f} MB")
    # # search data with "category = cat" and "split = test"
    # cats = dataset.items_metadata[
    #     (dataset.items_metadata["category"] == "cat")
    #     & (dataset.items_metadata["split"] == "test")
    # ]
    # start_time = time.time()
    # cats_sketch = [dataset[row.id] for _, row in cats[:100].iterrows()]
    # console.print(f"Loading 100 sketches in {time.time() - start_time}")
    # del dataset, cats

    # console.print()

    console.print("3. Load from CISLAB CDN", style="green")

    # tempfile can create a temporary directory
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = QuickDraw(root=tmpdir, cislab_source=True)
        # show brief information of the dataset
        console.print(dataset)
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        console.print(f"Current memory usage: {memory_mb:.2f} MB")
        dogs = dataset.items_metadata[
            (dataset.items_metadata["category"] == "dog")
            & (dataset.items_metadata["split"] == "train")
        ]
        start_time = time.time()
        dogs_sketch = [dataset[row.id] for _, row in dogs[:100].iterrows()]
        console.print(f"Loading 100 sketches in {time.time() - start_time}")
        del dataset, dogs

        console.print()
