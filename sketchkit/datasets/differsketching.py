import os
import shutil
import json
import requests
import zipfile
from tqdm import tqdm
from pathlib import Path as PathLibPath
import pandas as pd
import gdown

from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import download_with_tqdm, dir_md5, CISLAB_CDN, extract_files


def load_differsketching_json(json_path: str) -> Sketch:
    """Load a JSON sketch and convert it to a `Sketch` instance.

    Args:
        json_path (str): Path to a DifferSketching JSON file.

    Returns:
        Sketch: A `Sketch` object representing the vector sketch.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    strokes = data["strokes"]
    paths = []

    for stroke in strokes:
        points = stroke["path"]  # [[x,y], [x,y], ...]
        pressures = stroke.get("pressure", [None] * len(points))
        thickness = stroke.get("width", None)

        curves = []
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]

            p0 = pressures[i] if pressures else None
            p1 = pressures[i + 1] if pressures else None

            v_start = Vertex(x0, y0, pressure=p0, thickness=thickness)
            v_end = Vertex(x1, y1, pressure=p1, thickness=thickness)

            ctrl1 = Point(2/3*x0 + 1/3*x1, 2/3*y0 + 1/3*y1)
            ctrl2 = Point(1/3*x0 + 2/3*x1, 1/3*y0 + 2/3*y1)

            curve = Curve(v_start, v_end, ctrl1, ctrl2)
            curves.append(curve)

        if curves:
            path = Path(curves=curves)
            paths.append(path)

    return Sketch(paths=paths)


class DifferSketching(SketchDataset):
    """The DifferSketching dataset contains vector sketches with pressure and timing data.

    This dataset includes multiple drawing types (original, global, stroke, reg) 
    across various categories, with each sketch containing stroke sequences with
    pressure and timing information.

    Attributes:
        md5_sum (str): MD5 checksum for dataset integrity verification.
    """

    md5_sum = "97bea7240fa9d74716730f50406f9ba8"
    metadata = ["id", "file_path", "category", "type"]

    def __init__(
        self,
        root: str | PathLibPath | None = None,
        load_all: bool = False,
        cislab_source: bool = False,
    ):
        """Initialize the DifferSketching dataset.

        Args:
            root (str | PathLibPath | None): Root directory to store/load the dataset.
                If None, uses the default dataset directory.
            load_all (bool): If True, loads all data into memory for faster access.
                If False, loads data on-demand. Defaults to False.
            cislab_source (bool): If True, downloads from CISLAB CDN mirror.
                If False, downloads from Google Drive. Defaults to False.
        """
        super().__init__(root, load_all=load_all, cislab_source=cislab_source)

    def _check_integrity(self) -> bool:
        """Check the integrity of the cached dataset using MD5 checksum.

        Returns:
            bool: True if the dataset integrity is verified, False otherwise.
        """
        print(f"Checking integrity of cached {self.__class__.__name__} dataset...")
        src_dir = self._detect_src_dir()
        if src_dir is None:
            return False
        current_md5 = dir_md5(src_dir)
        return current_md5 == self.md5_sum

    def _download(self):
        """Download the DifferSketching dataset.

        Downloads the dataset archive from either Google Drive or CISLAB CDN mirror.
        Creates the root directory if it doesn't exist.

        Raises:
            Exception: If download fails.
        """
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

        archive_path = os.path.join(self.root, "differsketching.zip")
        try:
            if self.cislab_source:
                url = f"{CISLAB_CDN}/datasets/DifferSketching/differsketching.zip"
                download_with_tqdm(url, archive_path)
            else:
                gdown.download(id="1A_3RVc8Y4YdI7nhyM7tb-q7dQw4zTcCO", output=archive_path, quiet=False)
        except Exception as e:
            raise e

        try:
            extract_files(archive_path, self.root, remove_sourcefile=True)
        except Exception as e:
            raise e

    def _load_items_metadata(self):
        """Load and cache metadata for all items in the dataset.

        Creates a parquet file containing metadata for all sketches including
        category, type, file path, and global ID. If metadata cache exists, 
        loads from the cached file.

        The metadata DataFrame contains columns:
        - id: Global unique identifier across all sketches
        - file_path: Path to the JSON file
        - category: Drawing category name
        - type: Drawing type (original, global, stroke, reg)
        """
        if not os.path.exists(os.path.join(self.root, ".metadata.parquet")):
            src_dir = self._detect_src_dir()
            if src_dir is None:
                raise RuntimeError("Unable to locate dataset directory after extraction.")

            records = []
            cnt = 0
            base = PathLibPath(src_dir)
            for json_dir in base.rglob("*_json"):
                dir_name = json_dir.name
                type_name = dir_name[:-5] if dir_name.endswith("_json") else dir_name
                category = json_dir.parent.name
                for json_file in json_dir.glob("*.json"):
                    records.append({
                        "id": cnt,
                        "file_path": str(json_file),
                        "category": category,
                        "type": type_name,
                    })
                    cnt += 1

            if len(records) == 0:
                raise RuntimeError("No JSON files found. Please ensure dataset is downloaded correctly.")

            self.items_metadata = pd.DataFrame(records, columns=self.metadata)
            self.items_metadata.to_parquet(
                os.path.join(self.root, ".metadata.parquet"), compression="zstd"
            )
        else:
            items_metadata = pd.read_parquet(os.path.join(self.root, ".metadata.parquet"))
            if not set(items_metadata.columns.tolist()) == set(self.metadata):
                os.remove(os.path.join(self.root, ".metadata.parquet"))
                self._load_items_metadata()
                return
            self.items_metadata = items_metadata

        self.raw_data = [None] * len(self.items_metadata)

    def _load_all(self):
        """Load all sketch data into memory if load_all is enabled."""
        for idx in range(len(self.items_metadata)):
            if self.raw_data[idx] is None:
                self.raw_data[idx] = self._load_sketch(idx)

    def _load_sketch(self, idx: int) -> Sketch:
        """Load a single sketch by index."""
        row = self.items_metadata.iloc[idx]
        sketch = load_differsketching_json(row["file_path"])
        setattr(sketch, "category", row.get("category", None))
        setattr(sketch, "type", row.get("type", None))
        return sketch

    def __getitem__(self, idx: int) -> Sketch:
        """Get a sketch by index.

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
            self.raw_data[idx] = self._load_sketch(idx)
        return self.raw_data[idx]

    def _detect_src_dir(self) -> str | None:
        """Detect actual extracted dataset directory under root.

        Returns:
            str | None: Path to directory that contains *_json subfolders with JSON files.
        """
        # Preferred conventional folder name
        preferred = os.path.join(self.root, "differsketching")
        if os.path.isdir(preferred):
            return preferred
        # Otherwise scan immediate subdirectories
        candidates = [os.path.join(self.root, d) for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        # Include root itself in case archive extracted files directly
        candidates.append(self.root)
        for c in candidates:
            try:
                base = PathLibPath(c)
                for json_dir in base.rglob("*_json"):
                    if any(p.suffix.lower() == ".json" for p in json_dir.glob("*.json")):
                        return c
            except Exception:
                continue
        return None

    def extra_repr(self) -> str:
        """Return extra information for the string representation."""
        if len(self.items_metadata) == 0:
            return "No samples"
        categories = self.items_metadata["category"].nunique()
        types = self.items_metadata["type"].nunique()
        return f"Categories: {categories}, Types: {types}, Total: {len(self.items_metadata)} samples"


if __name__ == "__main__":
    # Default Load
    dataset = DifferSketching()

    # # Load all data to memory at once
    # dataset = DifferSketching(load_all=True)
    #
    # # Download from CISLAB CDN
    # dataset = DifferSketching(cislab_source=True)

    # Search items using metadata - filter by drawing types
    objects = dataset.items_metadata[
        (dataset.items_metadata["category"] == "Animal_Head")
        & (dataset.items_metadata["type"].isin(["original", "global"]))
    ]

    # Load sketches
    sketches = [dataset[row.id] for _, row in objects[:5].iterrows()]
    
    print(f"Successfully loaded {len(sketches)} sketches")
    if len(sketches) > 0:
        print(f"First sketch: {sketches[0].path_num} paths, {sketches[0].curve_num} curves")
        print(f"Category: {getattr(sketches[0], 'category', 'N/A')}")
        print(f"Type: {getattr(sketches[0], 'type', 'N/A')}")
    #     raster_image_png.save(outpath, "PNG")
