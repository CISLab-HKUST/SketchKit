import os
import shutil
import numpy as np
import pandas as pd
import json
import math

from sketchkit.core.sketch import Sketch, Path, Curve
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import dir_md5, CISLAB_CDN, download_with_wget, download_with_gdown, extract_files


def parse_stroke_data(stroke_datas, seg_labels, image_size=256):
    """Convert strokes into SketchKit's Sketch format.
    The coordinate values of points in this dataset are very small. So it is necessary to resize the sketches.

    Args:
        stroke_datas (list): Points for each path. Each path is a list (len=stroke_num) of [dx, dy, p1, p2].
            (p1, p2) = (1.0, 0.0), (0.0, 1.0), (0.0, 0.0).
            The first [dx, dy] is [0.0, 0.0], which should be discarded.
        seg_labels (list): A list of segmentation labels for each path.
        image_size (int): The size for resizing the sketches.

    Returns:
        tuple: A tuple containing:
            - path_list (list[Path]): List of Path objects, each containing
              stroke segments as cubic Bezier curves.
            - total_segment_num (int): Total number of stroke segments across
              all paths.
            - max_dim (float): Maximum canvas width/height.
    """
    path_list = []  # list of (N_strokes, 4, 2)
    total_segment_num = 0
    max_dim = 0.0

    xy_list = []
    for path_i, stroke_data_ in enumerate(stroke_datas):
        stroke_data = np.array(stroke_data_)
        xs = stroke_data[:, 0]  # (N_point + 1)
        ys = stroke_data[:, 1]  # (N_point + 1)
        xy = np.stack([xs, ys], axis=1)[1:, :]  # (N_point, 2), discard the first point [0, 0]
        xy_list.append(xy)

    xy_list_np = np.concatenate(xy_list, axis=0)
    min_xy = np.min(xy_list_np, axis=0, keepdims=True)  # (1, 2)
    max_xy = np.max(xy_list_np, axis=0, keepdims=True)  # (1, 2)
    max_dim_original = np.max(max_xy - min_xy)

    for path_i, xy_ in enumerate(xy_list):
        xy = (xy_ - min_xy) / max_dim_original * float(image_size)  # (N_point, 2)
        max_dim = max(math.ceil(np.max(xy)), max_dim)

        stroke_list = []  # list of (4, 2)
        for i in range(len(xy) - 1):
            p_start = xy[i]  # (2)
            p_end = xy[i + 1]  # (2)
            line = np.stack([p_start, p_end], axis=0)  # (2, 2)
            attribute_list = [{"group_id": seg_labels[path_i]},
                              {"group_id": seg_labels[path_i]}]
            cubic = Curve.from_line(line, attribute_list=attribute_list)  # (4, 2)
            stroke_list.append(cubic)
            total_segment_num += 1

        path_list.append(Path(stroke_list))

    assert max_dim == image_size
    return path_list, total_segment_num, max_dim


class SketchIME(SketchDataset):
    """SketchIME Dataset loader and interface.

    The SketchIME contains 209K drawings across 374 categories with segmentation annotation.
    Each drawing is represented as a sequence of strokes in stroke-3 format.

    Attributes:
        md5_sum (str): MD5 checksum for dataset integrity verification.

    References:
        - Original dataset: https://github.com/GuangmingZhu/SketchIME
        - Paper: "Sketch Input Method Editor: A Comprehensive Dataset and Methodology for Systematic Input Recognition" (https://dl.acm.org/doi/10.1145/3581783.3612115)
    """

    # MD5 Sum of the whole dir, you can set a list for zip files
    md5_sum = "e4aaaea8c37fedd79b361db32339ef59"
    # Metadata columns for each items
    metadata = ["id", "sub_id", "filename", "category", "split", "label_names"]
    TOTAL_PART_NUMBER = 139

    # def __init__(
    #     self,
    #     data_root: str | None = None,
    #     load_all: bool = False,
    #     cislab_source: bool = False,
    # ):
    #     super().__init__(data_root, load_all=load_all, cislab_source=cislab_source)

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
            self.root, "SketchIME", "cate_dir"
        )
        self.all_class_part_labels = ["ime-" + str(i) for i in range(self.TOTAL_PART_NUMBER)]  # such as "ime-11"

        if not os.path.exists(os.path.join(self.root, ".metadata.parquet")):
            self.all_categories = os.listdir(self.dataset_folder)
            self.all_categories.sort()

            items_metadata = []
            global_id = 0

            label_names = ",".join([str(i) for i in range(self.TOTAL_PART_NUMBER)])

            for category in self.all_categories:
                category_dir = os.path.join(self.dataset_folder, category)
                json_files = os.listdir(category_dir)
                json_files = [item for item in json_files if item.endswith(".json")]
                json_files.sort()

                sample_count = len(json_files)
                assert sample_count != 0

                valid_test_num = sample_count // (10 + 1 + 1)
                train_num = sample_count - valid_test_num - valid_test_num
                split_map = ["train"] * train_num + ["valid"] * valid_test_num + ["test"] * valid_test_num

                for sub_id in range(sample_count):
                    items_metadata.append(
                        {
                            "category": category,
                            "sub_id": sub_id,
                            "id": global_id,
                            "filename": json_files[sub_id],
                            "split": split_map[sub_id],
                            "label_names": label_names
                        }
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
            self.all_categories = sorted(
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
        """Download SketchIME Dataset from GitHub and unzip it.

        Raises:
            Exception: If download fails for any file.
        """
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

        zip_path = os.path.join(self.root, "SketchIME.zip")
        try:
            if self.cislab_source:
                url = (
                        f"{CISLAB_CDN}/datasets/{self.__class__.__name__}/"
                        + "SketchIME.zip"
                )
                download_with_wget(url, file_path=zip_path)
            else:
                download_with_gdown(
                    output_folder=self.root,
                    gdrive_id="1TbaK46IQvI6MCs0JpCcB1Fcx8Nr3i8g8",
                    filename="SketchIME.zip",
                )
        except Exception as e:
            raise e

        try:
            extract_files(
                file_path=zip_path,
                output_dir=self.root,
                remove_sourcefile=remove_zip,
            )
        except Exception as e:
            raise e

    def _load_all(self):
        """Load all sketch data into memory if load_all is enabled.

        Concatenates all sketch data from all categories and splits into a single
        numpy array for faster access. Only loads if self.load_all is True.
        """
        cnt = 0
        for category in self.all_categories:
            category_dir = os.path.join(self.root, "SketchIME", "cate_dir", category)
            json_files = os.listdir(category_dir)
            json_files = [item for item in json_files if item.endswith(".json")]
            json_files.sort()

            for json_file in json_files:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    seg_label = data["seg_label"]  # for each path: [seg_label, seg_label, ..., 139, ...]
                    seg_label = [item for item in seg_label if item != self.TOTAL_PART_NUMBER]
                    path_stroke_nums = data["stroke_num"]  # for each path: [stroke_num, stroke_num, ..., 0, ...]
                    path_stroke_nums = [item for item in path_stroke_nums if item != 0]
                    assert len(seg_label) == len(path_stroke_nums) == data["sketch_stroke_num"]

                    points_offsets_raw = data["points_offsets"]
                    # len=38; each is a list of len=256;
                    # each is [dx, dy, p1, p2], (p1, p2) = (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)
                    # the first [dx, dy] is [0.0, 0.0], which should be discarded

                    points_offsets = []
                    for path_i in range(data["sketch_stroke_num"]):
                        points_offsets.append(points_offsets_raw[path_i][:path_stroke_nums[path_i]])

                    self.raw_data[cnt] = [seg_label, points_offsets]
                    cnt += 1

    def _get_single(self, idx):
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
            category_dir = os.path.join(self.root, "SketchIME", "cate_dir", item_metadata["category"])
            json_file = os.path.join(category_dir, item_metadata["filename"])
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                seg_label = data["seg_label"]  # for each path: [seg_label, seg_label, ..., 139, ...]
                seg_label = [item for item in seg_label if item != self.TOTAL_PART_NUMBER]
                path_stroke_nums = data["stroke_num"]  # for each path: [stroke_num, stroke_num, ..., 0, ...]
                path_stroke_nums = [item for item in path_stroke_nums if item != 0]
                assert len(seg_label) == len(path_stroke_nums) == data["sketch_stroke_num"]

                points_offsets_raw = data["points_offsets"]
                # len=38; each is a list of len=256;
                # each is [dx, dy, p1, p2], (p1, p2) = (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)
                # the first [dx, dy] is [0.0, 0.0], which should be discarded

                points_offsets = []
                for path_i in range(data["sketch_stroke_num"]):
                    points_offsets.append(points_offsets_raw[path_i][:path_stroke_nums[path_i]])

                self.raw_data[idx] = [seg_label, points_offsets]

        seg_label, points_offsets = self.raw_data[idx]
        # seg_label: for each path: [seg_label, seg_label, ...]
        # points_offsets: points for each path. Each path is a list (len=stroke_num) of [dx, dy, p1, p2]
        #   (p1, p2) = (1.0, 0.0), (0.0, 1.0), (0.0, 0.0).
        #   The first [dx, dy] is [0.0, 0.0], which should be discarded.

        path_list, total_segment_num, max_dim = parse_stroke_data(points_offsets, seg_label)
        sketch = Sketch(height=max_dim, width=max_dim, paths=path_list)
        return sketch


if __name__ == "__main__":
    dataset = SketchIME()

    sketch_class = dataset.items_metadata[
        (dataset.items_metadata["category"] == "040") & (dataset.items_metadata["split"] == "test")
    ]
    sketches = [dataset[row.id] for _, row in sketch_class[:10].iterrows()]

    from sketchkit.renderer import CairoRenderer
    # Initialize a renderer for rendering
    renderer = CairoRenderer()

    for si, sketch in enumerate(sketches):
        raster_image = renderer.render(sketch)
        raster_image.save(f"sketch-{si}.png")
