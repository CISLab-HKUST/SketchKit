import os

import ndjson
import numpy as np
import pandas as pd

from sketchkit.core.sketch import Sketch, Path
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.transform import line_to_cubic
from sketchkit.utils.file import dir_md5, CISLAB_CDN, download_with_wget, extract_files


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


class SketchXPRIS(SketchDataset):
    """SketchX-PRIS-Dataset loader and interface.

    The SketchX-PRIS-Dataset contains 20000 drawings across 25 categories,
    collected by SKetchX (http://sketchx.eecs.qmul.ac.uk/) and PRIS (http://www.pris.net.cn/).
    Each drawing is represented as a sequence of strokes in stroke-3 format.

    Attributes:
        md5_sum (str): MD5 checksum for dataset integrity verification.

    References:
        - Original dataset: https://github.com/KeLi-SketchX/SketchX-PRIS-Dataset
        - Paper: "Universal Perceptual Grouping" (https://arxiv.org/abs/1808.02312)
    """

    # MD5 Sum of the whole dir, you can set a list for zip files
    md5_sum = "67ac28f552b857fc84de4f60775a13a0"
    # Metadata columns for each items
    metadata = ["id", "sub_id", "category"]

    def __init__(
        self,
        data_root: str | None = None,
        load_all: bool = False,
        cislab_source: bool = False,
    ):
        super().__init__(data_root, load_all=load_all, cislab_source=cislab_source)

    def _load_items_metadata(self):
        """Load and cache metadata for all items in the dataset.

        Creates a parquet file containing metadata for all sketches including
        category, group ID, global ID, and sub ID within category.
        If metadata cache exists, loads from the cached file.

        The metadata DataFrame contains columns:
        - category: Drawing category name
        - id: Global unique identifier across all sketches
        - sub_id: Identifier within the category and split
        """
        self.dataset_folder = os.path.join(
            self.root, "SketchX-PRIS-Dataset-master", "Perceptual Grouping"
        )
        if not os.path.exists(os.path.join(self.root, ".metadata.parquet")):
            self.group_folder = os.path.join(
                self.root, "SketchX-PRIS-Dataset-master", "Group ID"
            )
            ndjson_files = [
                f for f in os.listdir(self.dataset_folder) if f.endswith(".ndjson")
            ]
            self._all_categories = [os.path.splitext(f)[0] for f in ndjson_files]
            self._all_categories.sort()

            category_sample_count = {}
            for category in self._all_categories:
                ndjson_path = os.path.join(self.dataset_folder, f"{category}.ndjson")
                if not os.path.exists(ndjson_path):
                    print(
                        f"Warning: NDJSON file not found for category '{category}' (path: {ndjson_path})"
                    )
                    category_sample_count[category] = 0
                    continue

                with open(ndjson_path, "r", encoding="utf-8") as fp:
                    samples = ndjson.load(fp)[0]["train_data"]
                category_sample_count[category] = len(samples)
            items_metadata = []
            global_id = 0

            for category in self._all_categories:
                sample_count = category_sample_count[category]
                if sample_count == 0:
                    print(f"Skipping category '{category}' with 0 samples.")
                    continue

                for sub_id in range(sample_count):
                    items_metadata.append(
                        {"category": category, "sub_id": sub_id, "id": global_id}
                    )
                    global_id += 1

            self.items_metadata = pd.DataFrame(items_metadata)
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

    def _check_integrity(self) -> bool:
        """Check the integrity of the cached dataset using MD5 checksum.

        Returns:
            bool: True if the dataset integrity is verified, False otherwise.
        """
        print(f"Checking integrity of cached {self.__class__.__name__} dataset...")
        current_md5 = dir_md5(self.root)
        return current_md5 == self.md5_sum

    def _download(self, remove_zip: bool = True):
        """Download SketchX-PRIS Dataset from GitHub and unzip it.

        Raises:
            Exception: If download fails for any file.
        """
        if self.cislab_source:
            url = (
                f"{CISLAB_CDN}/datasets/{self.__class__.__name__}/" + "SketchX-PRIS.zip"
            )
        else:
            url = "https://github.com/KeLi-SketchX/SketchX-PRIS-Dataset/archive/refs/heads/master.zip"
        os.makedirs(self.root, exist_ok=True)
        zip_path = os.path.join(self.root, "SketchX-PRIS.zip")

        # Download zip
        if not os.path.exists(zip_path):
            print("Downloading SketchX-PRIS Dataset...")
            download_with_wget(
                url=url,
                file_path=zip_path,
                overwrite=True,
                pg_bar=True,
                desc="SketchX-PRIS",
            )
            print("Download finished.")
        else:
            print("Zip file already exists, skipping download.")

        # Extract zip
        extract_files(
            file_path=zip_path, output_dir=self.root, remove_sourcefile=remove_zip
        )
        print("MD5 Checksum:", dir_md5(self.root))

    def _load_all(self):
        """Load all sketch data into memory if load_all is enabled.

        Concatenates all sketch data from all categories and splits into a single
        numpy array for faster access. Only loads if self.load_all is True.
        """
        cnt = 0
        for category in self._all_categories:
            ndjson_path = os.path.join(
                self.root,
                "SketchX-PRIS-Dataset-master",
                "Perceptual Grouping",
                f"{category}.ndjson",
            )
            if not os.path.exists(ndjson_path):
                print(
                    f"Warning: NDJSON file not found for category '{category}' at {ndjson_path}"
                )
                continue
            with open(ndjson_path, "r", encoding="utf-8") as fp:
                try:
                    category_sketches = ndjson.load(fp)[0]["train_data"]
                    expected_count = self.items_metadata[
                        self.items_metadata["category"] == category
                    ].shape[0]
                    actual_count = len(category_sketches)

                    if actual_count != expected_count:
                        print(
                            f"Warning: Mismatch in sample count for '{category}': "
                            f"expected {expected_count}, got {actual_count}"
                        )

                    end_idx = cnt + actual_count
                    self.raw_data[cnt:end_idx] = category_sketches
                    cnt = end_idx

                except Exception as e:
                    print(
                        f"Error loading sketch data for category '{category}': {str(e)}"
                    )
                    continue

        loaded_count = sum(1 for item in self.raw_data if item is not None)
        print(
            f"Loaded {loaded_count}/{len(self.raw_data)} sketch samples into raw_data"
        )

        if loaded_count < len(self.raw_data):
            print(
                f"Warning: Some samples were not loaded (missing {len(self.raw_data) - loaded_count} items)"
            )

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        if self.raw_data[idx] is None:
            item_metadata = self.items_metadata.iloc[idx]
            category = item_metadata["category"]
            ndjson_path = os.path.join(self.dataset_folder, f"{category}.ndjson")
            with open(ndjson_path, "r", encoding="utf-8") as fp:
                samples = ndjson.load(fp)[0]["train_data"]
            cnt = self.items_metadata[
                self.items_metadata["category"] == item_metadata["category"]
            ]["id"].min()
            self.raw_data[cnt : cnt + len(samples)] = samples
            cnt += len(samples)
        single_sketch_stroke = self.raw_data[idx]
        path_list, total_segment_num = parse_stroke3_format(
            np.array(single_sketch_stroke)
        )
        sketch = Sketch(paths=path_list)
        return sketch


if __name__ == "__main__":
    dataset = SketchXPRIS(cislab_source=True)

    ice_cream = dataset.items_metadata[
        (dataset.items_metadata["category"] == "ice_cream")
    ]
    ice_cream_sketch = [dataset[row.id] for _, row in ice_cream[:100].iterrows()]

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer

    renderer = CairoRenderer(256, (1, 1, 1))
    raster_image = renderer.render(ice_cream_sketch[1])
    outpath = "test.png"
    if outpath is not None:
        raster_image_png = Image.fromarray(raster_image, "RGB")
        raster_image_png.save(outpath, "PNG")
