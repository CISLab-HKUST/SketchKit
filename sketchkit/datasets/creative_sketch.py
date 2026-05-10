import os
import shutil
import json
import numpy as np
from pathlib import Path as PathLibPath
import pandas as pd
from tqdm import tqdm
from rdp import rdp

from sketchkit.core.sketch import Sketch, Path, Curve
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import (
    download_with_gdown,
    download_with_wget,
    extract_files,
    file_md5,
    CISLAB_CDN,
)

COLORS_MAP = {
    "bird": {
        "initial": np.array([45, 169, 145]) / 255.0,
        "eye": np.array([243, 156, 18]) / 255.0,
        # "none": np.array([149, 165, 166]) / 255.0,
        "beak": np.array([211, 84, 0]) / 255.0,
        "body": np.array([41, 128, 185]) / 255.0,
        # "details": np.array([171, 190, 191]) / 255.0,
        "head": np.array([192, 57, 43]) / 255.0,
        "legs": np.array([142, 68, 173]) / 255.0,
        "mouth": np.array([39, 174, 96]) / 255.0,
        "tail": np.array([69, 85, 101]) / 255.0,
        "wings": np.array([127, 140, 141]) / 255.0,
    },
    "creature": {
        "initial": np.array([45, 169, 145]) / 255.0,
        "eye": np.array([243, 156, 18]) / 255.0,
        # "none": np.array([149, 165, 166]) / 255.0,
        "arms": np.array([211, 84, 0]) / 255.0,
        "beak": np.array([41, 128, 185]) / 255.0,
        "mouth": np.array([54, 153, 219]) / 255.0,
        "body": np.array([192, 57, 43]) / 255.0,
        "ears": np.array([142, 68, 173]) / 255.0,
        "feet": np.array([39, 174, 96]) / 255.0,
        "fin": np.array([69, 85, 101]) / 255.0,
        "hair": np.array([127, 140, 141]) / 255.0,
        "hands": np.array([45, 63, 81]) / 255.0,
        "head": np.array([241, 197, 17]) / 255.0,
        "horns": np.array([51, 205, 117]) / 255.0,
        "legs": np.array([232, 135, 50]) / 255.0,
        "nose": np.array([233, 90, 75]) / 255.0,
        "paws": np.array([160, 98, 186]) / 255.0,
        "tail": np.array([58, 78, 99]) / 255.0,
        "wings": np.array([198, 203, 207]) / 255.0,
        # "details": np.array([171, 190, 191]) / 255.0,
    }
}


class CreativeSketch(SketchDataset):
    """The Creative Sketch dataset contains vector sketches of birds and creatures with detailed part annotations.

    Attributes:
        md5_sum (str): MD5 checksum for dataset integrity verification.

    References:
        - Original dataset: https://songweige.github.io/projects/creative_sketech_generation/home.html
    """

    # MD5 Sum of the dataset
    md5_sum = "84c57a01499321bd6080b1f76754d709"

    # Metadata columns for each item
    metadata = ["id", "category", "sub_id", "is_good", "split", "label_names"]

    def __init__(
        self,
        root: str | PathLibPath | None = None,
        load_all: bool = False,
        cislab_source: bool = False,
        skip_integrity_check: bool = False,
        rdp_simplification: bool = False,
    ):
        """Initialize the Creative Sketch dataset.

        Args:
            root (str | PathLibPath | None): Root directory to store/load the dataset.
                If None, uses the default dataset directory.
            load_all (bool): If True, loads all data into memory for faster access.
                If False, loads data on-demand. Defaults to False.
            cislab_source (bool): If True, downloads from CISLAB CDN mirror.
                If False, downloads from original Google storage. Defaults to False.
            rdp_simplification (bool): If True, simplify the path points using RDP algorithm.
            skip_background (bool): If True, skip the "details" part label for background.
        """
        self.rdp_simplification = rdp_simplification
        super().__init__(root, load_all=load_all, cislab_source=cislab_source,
                         skip_integrity_check=skip_integrity_check)

    def _check_integrity(self) -> bool:
        """Check the integrity of the cached dataset.

        Returns:
            bool: True if the dataset integrity is verified, False otherwise.
        """
        print(f"Checking integrity of cached {self.__class__.__name__} dataset...")

        if not os.path.exists(os.path.join(self.root, "creative_sketch.zip")):
            print(
                f"creative_sketch.zip not found in {os.path.join(self.root, 'creative_sketch.zip')}."
            )
            return False

        if file_md5(os.path.join(self.root, "creative_sketch.zip")) != self.md5_sum:
            print(f"creative_sketch.zip md5 check failed.")
            return False

        bird_json_path = os.path.join(
            self.root, "raw_data_clean/creative_birds_json.txt"
        )
        creature_json_path = os.path.join(
            self.root, "raw_data_clean/creative_creatures_json.txt"
        )
        return os.path.exists(bird_json_path) and os.path.exists(creature_json_path)

    def _download(self):
        """Download the Creative Sketch dataset.

        Downloads the dataset from Google Drive and extracts it to the root directory.

        Raises:
            Exception: If download fails for any reason.
        """
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

        svg_zip_path = os.path.join(self.root, "creative_sketch.zip")
        try:
            if self.cislab_source:
                url = (
                    f"{CISLAB_CDN}/datasets/{self.__class__.__name__}/"
                    + "creative_sketch.zip"
                )
                download_with_wget(url, file_path=svg_zip_path)
            else:
                download_with_gdown(
                    output_folder=self.root,
                    gdrive_id="1Dstn3Cv2yAnxt1JXC0zPa1VPVDSz9gc-",
                    filename="creative_sketch.zip",
                )
        except Exception as e:
            raise e

        try:
            extract_files(
                file_path=svg_zip_path,
                output_dir=self.root,
                remove_sourcefile=False,
            )
        except Exception as e:
            raise e

    def _load_items_metadata(self):
        """Load and cache metadata for all items in the dataset.

        Creates a parquet file containing metadata for all sketches including
        category, sub_id, global ID, and whether the sample is good.
        If metadata cache exists, loads from the cached file.
        """

        bird_json_path = os.path.join(
            self.root, "raw_data_clean/creative_birds_json.txt"
        )
        creature_json_path = os.path.join(
            self.root, "raw_data_clean/creative_creatures_json.txt"
        )
        self.bird_json_base = json.loads(open(bird_json_path).read())
        self.creature_json_base = json.loads(open(creature_json_path).read())

        self.all_class_part_labels = []  # in "class-part" format, such as "dog-head"

        class_part_labels = list(COLORS_MAP["bird"].keys())
        class_part_labels.sort()
        self.all_class_part_labels += ["-".join(["bird", item]) for item in class_part_labels]
        label_names_bird = ";".join(class_part_labels)

        class_part_labels = list(COLORS_MAP["creature"].keys())
        class_part_labels.sort()
        self.all_class_part_labels += ["-".join(["creature", item]) for item in class_part_labels]
        label_names_creature = ";".join(class_part_labels)

        self.all_categories = list(COLORS_MAP.keys())
        self.all_categories.sort()

        if not os.path.exists(os.path.join(self.root, ".metadata.parquet")):
            items_metadata = pd.DataFrame(
                columns=["id", "category", "sub_id", "is_good", "split", "label_names"]
            )
            cnt = 0

            # Creative Bird
            valid_test_num = len(self.bird_json_base) // (10 + 1 + 1)
            train_num = len(self.bird_json_base) - valid_test_num - valid_test_num
            split_map = ["train"] * train_num + ["valid"] * valid_test_num + ["test"] * valid_test_num

            for sub_id, sketch in tqdm(
                enumerate(self.bird_json_base),
                total=len(self.bird_json_base),
                desc="Loading bird metadata",
            ):
                new_row = {
                    "id": cnt,
                    "category": "bird",
                    "sub_id": sub_id,
                    "is_good": sketch["good_sample"],
                    "split": split_map[sub_id],
                    "label_names": label_names_bird
                }
                items_metadata = pd.concat(
                    [items_metadata, pd.DataFrame([new_row])], ignore_index=True
                )
                cnt += 1

            # Creative Creatures
            valid_test_num = len(self.creature_json_base) // (10 + 1 + 1)
            train_num = len(self.creature_json_base) - valid_test_num - valid_test_num
            split_map = ["train"] * train_num + ["valid"] * valid_test_num + ["test"] * valid_test_num

            for sub_id, sketch in tqdm(
                enumerate(self.creature_json_base),
                total=len(self.creature_json_base),
                desc="Loading creature metadata",
            ):
                new_row = {
                    "id": cnt,
                    "category": "creature",
                    "sub_id": sub_id,
                    "is_good": sketch["good_sample"],
                    "split": split_map[sub_id],
                    "label_names": label_names_creature
                }
                items_metadata = pd.concat(
                    [items_metadata, pd.DataFrame([new_row])], ignore_index=True
                )
                cnt += 1
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
                self._load_items_metadata()
                return
            self.items_metadata = items_metadata

        self.raw_data = [None] * len(self.items_metadata)

    def _load_all(self):
        """Load all sketch data into memory if load_all is enabled.

        Loads all sketch data from the JSON files into memory for faster access.
        Only loads if self.load_all is True.
        """
        for idx in tqdm(
            range(len(self.items_metadata)),
            desc="Loading all Creative Sketch data to memory",
        ):
            if self.raw_data[idx] is None:
                self.raw_data[idx] = self._load_sketch_data(idx)

    def _load_sketch_data(self, idx):
        """Load a single sketch data by index.

        Args:
            idx (int): Index of the sketch to load.

        Returns:
            list: List of paths for the sketch.
        """
        category, orig_idx = self.items_metadata.iloc[idx][["category", "sub_id"]]
        if category == "bird":
            single_sketch_json = self.bird_json_base[orig_idx]
            colors = COLORS_MAP["bird"]
        else:
            single_sketch_json = self.creature_json_base[orig_idx]
            colors = COLORS_MAP["creature"]
        class_part_labels = list(colors.keys())
        class_part_labels.sort()

        path_list = []
        for j, component in enumerate(single_sketch_json["all_strokes"]):
            stroke_label = single_sketch_json["partsUsed"][j]
            # stroke_color = colors[stroke_label]

            if stroke_label not in COLORS_MAP[category].keys():
                continue

            curves = []
            for path in component:
                if len(path) == 0:  # skip the empty path
                    continue

                stroke_list =[path[0][:2]] + [stroke[2:4] for stroke in path]  # A list (len=N) of (2)
                if self.rdp_simplification:
                    stroke_list = rdp(stroke_list, epsilon=1.0)  # Bigger epsilon, the more simplified.

                for seg_i in range(len(stroke_list) - 1):
                    p_start = [stroke_list[seg_i][0], stroke_list[seg_i][1]]
                    p_end = [stroke_list[seg_i + 1][0], stroke_list[seg_i + 1][1]]
                    line = np.stack([p_start, p_end], axis=0)
                    attribute_list = [{"group_id": class_part_labels.index(stroke_label)},
                                      {"group_id": class_part_labels.index(stroke_label)}]
                    cubic = Curve.from_line(line, attribute_list=attribute_list)
                    # cubic._set_endpoint_attribute("color", stroke_color)
                    curves.append(cubic)
            path_list.append(Path(curves=curves))
        return Sketch(512, 512, path_list)

    def _get_single(self, idx: int) -> Sketch:
        """Get a sketch by index.

        If a sketch not in memory, load it from disk.

        Args:
            idx (int): Index of the sketch to retrieve.

        Returns:
            Sketch: A Sketch object containing the drawing data as paths.

        Raises:
            IndexError: If index is out of range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        if self.raw_data[idx] is None:
            self.raw_data[idx] = self._load_sketch_data(idx)

        return self.raw_data[idx]

    def extra_repr(self) -> str:
        """Return extra information for the string representation.

        Returns:
            str: Additional information to include in __repr__.
        """
        num_birds = len(self.items_metadata[self.items_metadata["category"] == "bird"])
        num_creatures = len(
            self.items_metadata[self.items_metadata["category"] == "creature"]
        )
        num_good = len(self.items_metadata[self.items_metadata["is_good"] == True])

        return (
            f"Total samples: {len(self)}\n"
            f"Bird samples: {num_birds}\n"
            f"Creature samples: {num_creatures}\n"
            f"Good samples: {num_good}"
            f"Bad samples: {len(self) - num_good}"
        )


if __name__ == "__main__":
    dataset = CreativeSketch(cislab_source=True, skip_integrity_check=True,
                             # rdp_simplification=True
                             )

    birds = dataset.items_metadata[
        (dataset.items_metadata["category"] == "bird")
    ]
    bird_sketches = [dataset[row.id] for _, row in birds[:10].iterrows()]

    from sketchkit.renderer import CairoRenderer

    # Initialize a renderer for rendering
    renderer = CairoRenderer()

    for si, sketch in enumerate(bird_sketches):
        raster_image = renderer.render(sketch)
        raster_image.save(f"sketch-{si}.png")
