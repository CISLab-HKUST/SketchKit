"""DifferSketching dataset utilities.

This module provides helper functions to download the DifferSketching dataset
and convert its JSON sketches into the SketchKit `Sketch` format.

"""

import os
import requests
import zipfile
from tqdm import tqdm
import json
from pathlib import Path as SysPath
import pandas as pd
import gdown
from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import download_with_tqdm, dir_md5, CISLAB_CDN, extract_files

class DifferSketching:
    """DifferSketching dataset loader.

    This class indexes all JSON sketches under the DifferSketching dataset
    directory and provides random access via ``__getitem__`` similar to
    ``CreativeSketch``.

    Attributes:
        base_path (str): Root directory of the extracted dataset.
        items (list[dict]): Indexed items with keys ``path``, ``category``,
            and ``type``.

    Example:
        >>> data = DifferSketching(output_folder="data/differsketching")
        >>> len(data)
        123
        >>> s = data[0]
        >>> isinstance(s, Sketch)
        True
        >>> s.category, s.type
        ('Animal_Head', 'original')
    """

    def __init__(
        self,
        output_folder: str = "data/differsketching",
        allowed_types: tuple[str, ...] | None = None,
    ):
        """Initialize the dataset and build an index.

        Args:
            output_folder (str): Directory where the dataset is (or will be)
                downloaded and extracted.
            allowed_types (tuple[str, ...] | None): Optional whitelist of types
                (e.g., ("original", "global", "stroke", "reg")). If None,
                include all types found.
        """
        self.base_path = download_differsketching(output_folder)
        self.items = []

        base = SysPath(self.base_path)
        for json_dir in base.rglob("*_json"):
            dir_name = json_dir.name
            type_name = dir_name[:-5] if dir_name.endswith("_json") else dir_name
            if allowed_types is not None and type_name not in allowed_types:
                continue
            category = json_dir.parent.name
            for json_file in json_dir.glob("*.json"):
                self.items.append(
                    {
                        "path": str(json_file),
                        "category": category,
                        "type": type_name,
                    }
                )

    def __len__(self):
        """Return the number of indexed sketches.

        Returns:
            int: Total number of sketches.
        """
        return len(self.items)

    def __getitem__(self, idx: int) -> Sketch:
        """Load a sketch by index and attach metadata attributes.

        Args:
            idx (int): Index of the sketch.

        Returns:
            Sketch: Loaded sketch with ``category`` and ``type`` attributes.

        Raises:
            IndexError: If ``idx`` is out of range.
        """
        if idx < 0 or idx >= len(self.items):
            raise IndexError("Index out of range")
        item = self.items[idx]
        sketch = load_differsketching_json(item["path"])
        setattr(sketch, "category", item["category"])
        setattr(sketch, "type", item["type"])
        return sketch


class DifferSketchingDataset(SketchDataset):
    """DifferSketching dataset (JSON sequence data).

    This class conforms to SketchKit's `SketchDataset` interface, supporting
    integrity checks, mirrored downloads, metadata caching, lazy loading, and
    an optional eager `_load_all` path.

    Attributes:
        md5_sum (str): Expected MD5 of the extracted dataset directory. Replace
            with the correct value once confirmed by admins.
        gdrive_id (str): Google Drive file id for the dataset archive.
        filename (str): Archive filename.
        metadata (list[str]): Column names for items metadata.
    """

    md5_sum: str | None = "97bea7240fa9d74716730f50406f9ba8"
    gdrive_id: str = "1A_3RVc8Y4YdI7nhyM7tb-q7dQw4zTcCO"
    filename: str = "differsketching.zip"
    metadata = ["id", "file_path", "category", "type"]

    def __init__(
        self,
        root: str | None = None,
        load_all: bool = False,
        cislab_source: bool = False,
        allowed_types: tuple[str, ...] | None = None,
    ):
        self.allowed_types = allowed_types
        super().__init__(root=root, load_all=load_all, cislab_source=cislab_source)

    def _check_integrity(self) -> bool:
        """Verify dataset cache integrity using directory MD5 if available.

        Returns:
            bool: True if the dataset looks valid.
        """
        src_dir = self._detect_src_dir()
        if src_dir is None:
            return False
        if self.md5_sum is None:
            # Fallback: basic existence check without strict MD5 (until provided)
            return True
        current_md5 = dir_md5(src_dir)
        return current_md5 == self.md5_sum

    def _download(self):
        """Download and extract DifferSketching from CISLAB or Google Drive."""
        # Clean root for fresh extraction
        if os.path.exists(self.root):
            # do not remove root; only remove existing extracted dir/archive if any
            pass
        os.makedirs(self.root, exist_ok=True)

        archive_path = os.path.join(self.root, self.filename)
        try:
            if self.cislab_source:
                url = f"{CISLAB_CDN}/datasets/DifferSketching/{self.filename}"
                print(f"Attempting download from CISLAB: {url}")
                download_with_tqdm(url, archive_path, desc="Downloading from CISLAB CDN")
                print("CISLAB download completed successfully!")
            else:
                print("Using gdown for Google Drive download...")
                gdown.download(id=self.gdrive_id, output=archive_path, quiet=False)
                print("Google Drive download completed successfully!")
        except Exception as e:
            print("Download failed.")
            raise e

        try:
            extract_files(archive_path, self.root, remove_sourcefile=True)
        except Exception as e:
            print("Extraction failed.")
            raise e

        src_dir = self._detect_src_dir()
        if src_dir and self.md5_sum is not None:
            print("MD5 Checksum:", dir_md5(src_dir))

    def _load_items_metadata(self):
        """Build items metadata DataFrame and persist to parquet."""
        self.src_dir = self._detect_src_dir()
        if self.src_dir is None:
            raise RuntimeError("Unable to locate dataset directory after extraction.")
        md_path = os.path.join(self.root, ".metadata.parquet")

        if not os.path.exists(md_path):
            records = []
            cnt = 0
            base = SysPath(self.src_dir)
            for json_dir in base.rglob("*_json"):
                dir_name = json_dir.name
                type_name = dir_name[:-5] if dir_name.endswith("_json") else dir_name
                if self.allowed_types is not None and type_name not in self.allowed_types:
                    continue
                category = json_dir.parent.name
                for json_file in json_dir.glob("*.json"):
                    records.append(
                        {
                            "id": cnt,
                            "file_path": str(json_file),
                            "category": category,
                            "type": type_name,
                        }
                    )
                    cnt += 1

            if len(records) == 0:
                raise RuntimeError("No JSON files found. Please ensure dataset is downloaded correctly.")

            self.items_metadata = pd.DataFrame(records, columns=self.metadata)
            self.items_metadata.to_parquet(md_path, compression="zstd")
        else:
            items_metadata = pd.read_parquet(md_path)
            # Ensure required columns include at least 'id'
            if "id" not in items_metadata.columns:
                os.remove(md_path)
                self._load_items_metadata()
                return
            self.items_metadata = items_metadata

        self.raw_data = [None] * len(self.items_metadata)

    def _load_all(self):
        """Eagerly load all sketches into memory."""
        for idx in range(len(self.items_metadata)):
            if self.raw_data[idx] is None:
                self.raw_data[idx] = self._load_sketch(idx)

    def _load_sketch(self, idx: int) -> Sketch:
        row = self.items_metadata.iloc[idx]
        sketch = load_differsketching_json(row["file_path"])
        setattr(sketch, "category", row.get("category", None))
        setattr(sketch, "type", row.get("type", None))
        return sketch

    def __getitem__(self, idx: int) -> Sketch:
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
                base = SysPath(c)
                for json_dir in base.rglob("*_json"):
                    if any(p.suffix.lower() == ".json" for p in json_dir.glob("*.json")):
                        return c
            except Exception:
                continue
        return None

def download_differsketching(output_folder):
    """Download and extract the DifferSketching dataset.

    Args:
        output_folder (str): Directory to store the downloaded archive and
            extracted dataset.

    Returns:
        str: Absolute path to the extracted dataset directory.

    Notes:
        This function uses a Google Drive export URL. For large files,
        token-confirmation may be required and could fail in some environments.
        Consider replacing with `gdown` if robustness is required.
    """
    os.makedirs(output_folder, exist_ok=True)

    url = "https://drive.google.com/uc?id=1A_3RVc8Y4YdI7nhyM7tb-q7dQw4zTcCO&export=download"
    zip_path = os.path.join(output_folder, "differsketching.zip")
    extracted_folder = os.path.join(output_folder, "differsketching")

    if os.path.exists(extracted_folder):
        print("Differsketching already exists and is extracted. Skipping download and extraction.")
        return extracted_folder

    if not os.path.exists(zip_path):
        print("Downloading Differsketching dataset...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    else:
        print("zip already exists. Skipping download.")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_folder)

    print("Extracting done.！")
    return extracted_folder


def load_differsketching_json(json_path):
    """Load a JSON sketch and convert it to a `Sketch` instance.

    Args:
        json_path (str): Path to a DifferSketching JSON file.

    Returns:
        Sketch: A `Sketch` object representing the vector sketch.

    Raises:
        ValueError: If the JSON file content is malformed.
        OSError: If the file cannot be opened or read.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    strokes = data["strokes"]
    paths = []

    for stroke in strokes:
        points = stroke["path"]               # [[x,y], [x,y], ...]
        pressures = stroke.get("pressure", [None] * len(points))
        thickness = stroke.get("width", None)

        curves = []
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]

            p0 = pressures[i] if pressures else None
            p1 = pressures[i + 1] if pressures else None

            v_start = Vertex(x0, y0, pressure=p0, thickness=thickness)
            v_end   = Vertex(x1, y1, pressure=p1, thickness=thickness)


            ctrl1 = Point(2/3*x0 + 1/3*x1, 2/3*y0 + 1/3*y1)
            ctrl2 = Point(1/3*x0 + 2/3*x1, 1/3*y0 + 2/3*y1)

            curve = Curve(v_start, v_end, ctrl1, ctrl2)
            curves.append(curve)

        if curves:
            path = Path(curves=curves)
            paths.append(path)

    return Sketch(paths=paths)

def convert_differsketching(base_folder):
    """Convert all JSON sketches under a base folder into `Sketch` objects.

    This function scans recursively for directories matching the pattern
    "*_json" and converts any "*.json" files within to `Sketch` objects.

    Args:
        base_folder (str): Base directory that contains per-category
            subdirectories ending with "_json".

    Returns:
        list[Sketch]: A list of converted `Sketch` objects. Each `Sketch`
            instance will have extra attributes:
            - `category`: Category name from parent folder (e.g., `Animal_Head`).
            - `type`: Type name from the `*_json` folder (e.g., `original`,
              `global`, `stroke`, `reg`).

    Notes:
        This is a utility/convenience function intended mainly for batch
        conversion and debugging. For production usage, consider wrapping
        the dataset in a class that conforms to `SketchDataset`.
    """
    all_sketches = []
    base = SysPath(base_folder)

    for json_dir in base.rglob("*_json"):
        # Dataset layout:
        # [Category]/[Type]_json/*.json
        # - category should be the name of the parent directory
        # - type is derived from the json_dir name without the `_json` suffix
        dir_name = json_dir.name
        type_name = dir_name[:-5] if dir_name.endswith("_json") else dir_name
        category = json_dir.parent.name

        for json_file in json_dir.glob("*.json"):
            try:
                sketch = load_differsketching_json(json_file)
                # Attach category and type attributes to Sketch
                try:
                    setattr(sketch, "category", category)
                    setattr(sketch, "type", type_name)
                except Exception:
                    pass
                print(f"Converted: {json_file} → {sketch.path_num} paths, {sketch.curve_num} curves, category={category}, type={type_name}")
                all_sketches.append(sketch)
            except Exception as e:
                print(f"Failed to convert {json_file}: {e}")

    return all_sketches

if __name__ == "__main__":
    print("Initializing DifferSketchingDataset...")
    ds = DifferSketchingDataset(load_all=False, cislab_source=False)
    print("Number of samples:", len(ds))
    if len(ds) > 0:
        s0 = ds[0]
        print("First sample:")
        print("  paths=", s0.path_num, "curves=", s0.curve_num)
        print("  category=", getattr(s0, "category", None))
        print("  type=", getattr(s0, "type", None))
    # Print current directory MD5 to facilitate hardcoding expected md5_sum
    src_dir = ds._detect_src_dir()
    if os.path.isdir(src_dir):
        try:
            current_md5 = dir_md5(src_dir)
            print("Current directory MD5:", current_md5)
            print("Copy this value to DifferSketchingDataset.md5_sum for strict integrity checks.")
        except Exception as e:
            print("Failed to compute directory MD5:", e)
