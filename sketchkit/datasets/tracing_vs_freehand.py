import os
import shutil

import xml.etree.ElementTree as ET
from pathlib import Path as PathLibPath

import pandas as pd
from svg.path import parse_path, path
from tqdm import tqdm

from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import download_with_wget, dir_md5, CISLAB_CDN, extract_files


def _parse_svg_to_sketch(svg_path: str) -> Sketch:
    """Parses an SVG file and converts its content into a Sketch object.

    Args:
        svg_path (str): The file path to the SVG file to be parsed.

    Returns:
        Sketch: An object representing the parsed sketch. Returns an empty Sketch
            if the file cannot be parsed or is not found.
    """
    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()
        namespaces = {"svg": "http://www.w3.org/2000/svg"}

        sketch_paths = []
        for elem in root.findall("svg:path", namespaces):
            path_d = elem.attrib.get("d", "")
            if not path_d:
                continue

            parsed = parse_path(path_d)
            curves = []
            for segment in parsed:
                if isinstance(segment, path.CubicBezier):
                    p_start = Vertex(segment.start.real, segment.start.imag)
                    p_end = Vertex(segment.end.real, segment.end.imag)
                    p_ctrl1 = Point(segment.control1.real, segment.control1.imag)
                    p_ctrl2 = Point(segment.control2.real, segment.control2.imag)
                    curves.append(Curve(p_start, p_end, p_ctrl1, p_ctrl2))
                elif isinstance(segment, path.Line):
                    # Convert line to an equivalent cubic Bézier curve.
                    start, end = segment.start, segment.end
                    p_start_v = Vertex(start.real, start.imag)
                    p_end_v = Vertex(end.real, end.imag)
                    c1 = start * (2 / 3) + end * (1 / 3)
                    c2 = start * (1 / 3) + end * (2 / 3)
                    p_ctrl1 = Point(c1.real, c1.imag)
                    p_ctrl2 = Point(c2.real, c2.imag)
                    curves.append(Curve(p_start_v, p_end_v, p_ctrl1, p_ctrl2))

            if curves:
                sketch_paths.append(Path(curves=curves))

        return Sketch(paths=sketch_paths)
    except (ET.ParseError, FileNotFoundError):
        print(f"Warning: Failed to parse or find SVG file: {svg_path}")
        return Sketch(paths=[])


class TracingVsFreehand(SketchDataset):
    """Tracing-vs-Freehand dataset loader.

    This dataset contains sketches categorized as freehand drawings, registered
    drawings, and tracings, stored in SVG format. This loader handles automatic
    downloading, integrity checking, and parsing of these SVG files.

    Attributes:
        md5_sum (str): MD5 checksum for the extracted dataset directory.
        URL (str): Download URL for the dataset zip file.
    """

    md5_sum = "cdf5ce49867d96e1bd0bc0c7198fb95b"
    URL = f"{CISLAB_CDN}/datasets/TracingVsFreehand/sketch.zip"
    metadata = ["id", "file_path", "sketch_type", "category", "file_id"]

    def __init__(
        self,
        root: PathLibPath | str | None = None,
        load_all: bool = False,
        cislab_source: bool = True,
    ):
        """Initializes the Tracing-vs-Freehand dataset loader.

        Args:
            root (Optional[PathLibPath | str]): Root directory for the dataset.
                If None, a default cache path is used.
            load_all (bool): If True, loads all data into memory at init.
            cislab_source (bool): If True, uses the CISLAB CDN. This dataset is
                only available from this source.
        """
        super().__init__(root, load_all=load_all, cislab_source=cislab_source)

    def _check_integrity(self) -> bool:
        """Checks if the dataset is present and uncorrupted.

        Returns:
            bool: True if the dataset's integrity is verified, False otherwise.
        """
        print(f"Checking integrity of cached {self.__class__.__name__} dataset...")
        current_md5 = dir_md5(self.root)
        return current_md5 == self.md5_sum

    def _download(self):
        """Downloads and extracts the dataset from the source URL.

        Raises:
            RuntimeError: If the download or extraction process fails.
        """
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

        zip_path = os.path.join(self.root, "sketch.zip")

        try:
            download_with_wget(self.URL, zip_path, desc="Downloading dataset")
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}")

        extract_files(zip_path, self.root, remove_sourcefile=True)
        print("MD5 Checksum:", dir_md5(self.root))

    def _load_items_metadata(self):
        """Scans the dataset directory to create and cache metadata.

        This method generates a pandas DataFrame with metadata for each sketch
        and caches it as a .parquet file for fast subsequent loading.

        Raises:
            FileNotFoundError: If no SVG files are found after scanning.
        """
        metadata_cache_path = os.path.join(self.root, ".metadata.parquet")
        dataset_path = os.path.join(self.root, "data", "svg")

        if os.path.exists(metadata_cache_path):
            items_metadata = pd.read_parquet(
                os.path.join(self.root, ".metadata.parquet")
            )
            if not set(items_metadata.columns.tolist()) == set(self.metadata):
                os.remove(os.path.join(self.root, ".metadata.parquet"))
                return self._load_items_metadata()
            self.items_metadata = items_metadata
        else:
            print("Metadata cache not found. Scanning files...")
            metadata_list = []
            sketch_types = ["drawings", "drawings_registered", "tracings"]

            for sketch_type in sketch_types:
                type_dir = os.path.join(dataset_path, sketch_type)
                if not os.path.isdir(type_dir):
                    print(f"Warning: Directory '{sketch_type}' not found, skipping.")
                    continue

                files = sorted(os.listdir(type_dir))
                cnt = 0
                for filename in tqdm(files, desc=f"Scanning {sketch_type}"):
                    if filename.endswith(".svg"):
                        file_path = os.path.join(type_dir, filename)
                        parts = os.path.splitext(filename)[0].rsplit("_", 1)
                        category = parts[0]
                        file_id = parts[1] if len(parts) > 1 else "unknown"

                        metadata_list.append(
                            {
                                "id": cnt,
                                "file_path": file_path,
                                "sketch_type": sketch_type,
                                "category": category,
                                "file_id": file_id,
                            }
                        )
                        cnt += 1

            if not metadata_list:
                raise FileNotFoundError(
                    f"No SVG files found in {dataset_path}. "
                    "The dataset might be corrupted."
                )

            self.items_metadata = pd.DataFrame(metadata_list)
            self.items_metadata["id"] = range(len(self.items_metadata))
            self.items_metadata.to_parquet(metadata_cache_path, compression="zstd")

        self.raw_data = [None] * len(self.items_metadata)

    def _load_all(self):
        """Loads all sketch data into memory if `load_all` is True."""

        print("Loading all SVG data into memory...")
        for idx, row in tqdm(self.items_metadata.iterrows(), total=len(self)):
            try:
                # Parsing directly is more memory-efficient than storing text.
                self.raw_data[idx] = _parse_svg_to_sketch(row["file_path"])
            except Exception as e:
                print(f"Warning: Could not load {row['file_path']}. Error: {e}")
                self.raw_data[idx] = Sketch(paths=[])

    def __getitem__(self, idx: int) -> Sketch:
        """Retrieves a single sketch from the dataset by its index.

        Args:
            idx (int): The index of the sketch to retrieve.

        Returns:
            Sketch: The sketch object at the specified index.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if not 0 <= idx < len(self):
            raise IndexError("Index out of range")

        if self.raw_data[idx] is None:
            file_path = self.items_metadata.iloc[idx]["file_path"]
            self.raw_data[idx] = _parse_svg_to_sketch(file_path)

        return self.raw_data[idx]

    def extra_repr(self) -> str:
        """Returns a string with extra information about the dataset.

        Returns:
            str: A string containing dataset statistics by sketch type.
        """
        if not hasattr(self, "items_metadata") or self.items_metadata.empty:
            return "Metadata not loaded."
        counts = self.items_metadata["sketch_type"].value_counts()
        return (
            f"Sketch Types:\n"
            f"    drawings: {counts.get('drawings', 0)}\n"
            f"    drawings_registered: {counts.get('drawings_registered', 0)}\n"
            f"    tracings: {counts.get('tracings', 0)}"
        )


if __name__ == "__main__":
    import psutil
    from rich.console import Console
    import time

    console = Console()

    console.print("TracingVsFreehand Dataset Test", style="bold blue")
    console.print()

    # console.print("1. Default Load", style="green")
    # dataset = TracingVsFreehand()
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
    # dataset = TracingVsFreehand(load_all=True)
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
        dataset = TracingVsFreehand(cislab_source=True)
        # show brief information of the dataset
        console.print(dataset)
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        console.print(f"Current memory usage: {memory_mb:.2f} MB")
        dogs = dataset.items_metadata[
            (dataset.items_metadata["category"] == "drawings")
        ]
        start_time = time.time()
        dogs_sketch = [dataset[row.id] for _, row in dogs[:100].iterrows()]
        console.print(f"Loading 100 sketches in {time.time() - start_time}")
        console.print(dataset.items_metadata[:5])
        del dataset, dogs
        console.print()
