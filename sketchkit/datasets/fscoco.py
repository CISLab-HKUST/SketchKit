import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sketchkit.core.sketch import Sketch, Path as SKPath, Curve
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import download_with_wget, dir_md5, CISLAB_CDN, extract_files
from sketchkit.utils.transform import line_to_cubic


class FSCOCO(SketchDataset):
    """
    FS-COCO dataset loader for vector sketches stored in .npy format.

    This class provides functionality to download, verify, and load the FS-COCO dataset.
    The dataset consists of vector sketches, which are converted into cubic Bezier curves.

    Attributes:
        md5_sum (str): MD5 checksum for verifying dataset integrity.
        metadata (list): List of metadata columns for the dataset.
    """

    md5_sum: str = "5e5c08c2e6877b164b5d04f3ce8b9f89"
    metadata = ["id", "shard", "path"]

    def __init__(
        self,
        root: str | Path | None = None,
        load_all: bool = False,
        cislab_source: bool = False,
    ):
        """
        Initializes the FSCOCO dataset loader.

        Args:
            root (str | Path | None): Root directory for the dataset. The dataset will be extracted under `fscoco/`.
            load_all (bool): Whether to preload all .npy files into memory. Defaults to False.
            cislab_source (bool): If True, use the CISLAB mirror for downloading. Otherwise, use the official source.
        """
        super().__init__(root, load_all=load_all, cislab_source=cislab_source)

    def _check_integrity(self) -> bool:
        """
        Checks the integrity of the dataset directory.

        Returns:
            bool: True if the dataset is complete, False otherwise.
        """
        root = Path(self.root)
        data_dir = root / "fscoco" / "vector_sketches"
        if not data_dir.exists():
            return False
        current_md5 = dir_md5(root / "fscoco")
        return current_md5 == self.md5_sum

    def _download(self):
        """
        Downloads and extracts the FS-COCO dataset if integrity checks fail.

        The dataset is downloaded from either the CISLAB mirror or the official source.
        """

        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

        root = Path(self.root)

        archive = root / "fscoco.tar.gz"

        url = (
            f"{CISLAB_CDN}/datasets/FSCOCO/fscoco.tar.gz"
            if self.cislab_source
            else "http://cvssp.org/data/fscoco/fscoco.tar.gz"
        )

        print(f"[FS-COCO] Downloading from: {url}")
        download_with_wget(
            url, file_path=str(archive), desc="Downloading FS-COCO", pg_bar=True
        )

        print("[FS-COCO] Extracting...")
        extract_files(archive, root, remove_sourcefile=True)

        try:
            print("[FS-COCO] Directory MD5:", dir_md5(root))
        except Exception:
            pass

    def _load_items_metadata(self):
        """
        Loads metadata for the dataset by scanning the directory structure.

        Metadata includes:
        - id: A globally unique identifier for each sketch.
        - shard: The shard number (1 to 100).
        - path: The relative path to the .npy file.

        Metadata is cached in `.metadata.parquet` for faster subsequent loads.
        """
        meta_path = Path(self.root) / ".metadata.parquet"
        data_dir = Path(self.root) / "fscoco" / "vector_sketches"

        if meta_path.exists():
            df = pd.read_parquet(meta_path)
            if set(df.columns.tolist()) == set(self.metadata):
                self.items_metadata = df
                self.raw_data = [None] * len(self.items_metadata)
                return
            else:
                meta_path.unlink(missing_ok=True)

        rows: list[dict] = []
        idx = 0
        for shard in range(1, 101):
            sub = data_dir / f"{shard}"
            if not sub.exists():
                continue
            for npy_file in sorted(sub.glob("*.npy")):
                rows.append(
                    {
                        "id": idx,
                        "shard": shard,
                        "path": str(npy_file.relative_to(self.root)),
                    }
                )
                idx += 1

        df = pd.DataFrame(rows, columns=self.metadata)
        self.items_metadata = df
        self.items_metadata.to_parquet(meta_path, compression="zstd")
        self.raw_data = [None] * len(self.items_metadata)

    def _load_all(self):
        """
        Preloads all .npy files into memory.

        This method is optional and is typically used when `load_all` is set to True.
        """
        for i in tqdm(range(len(self.items_metadata)), desc="Loading all FS-COCO"):
            self.raw_data[i] = self._load_single_array(i)

    def __getitem__(self, idx: int) -> Sketch:
        """
        Retrieves a single sketch by index.

        Args:
            idx (int): Index of the sketch to retrieve.

        Returns:
            Sketch: The sketch object corresponding to the given index.

        Raises:
            IndexError: If the index is out of range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        if self.raw_data[idx] is None:
            self.raw_data[idx] = self._load_single_array(idx)

        return self._array_to_sketch(self.raw_data[idx])

    def _load_single_array(self, idx: int) -> np.ndarray:
        """
        Loads a single .npy file into memory.

        Args:
            idx (int): Index of the .npy file to load.

        Returns:
            np.ndarray: The loaded array.
        """
        rel = self.items_metadata.iloc[idx]["path"]
        path = Path(self.root) / rel
        return np.load(str(path))  # (N,3) = x,y,flag

    @staticmethod
    def _array_to_sketch(data: np.ndarray) -> Sketch:
        """
        Converts an array of points into a Sketch object.

        Args:
            data (np.ndarray): Array of shape (N, 3), where each row is (x, y, flag).

        Returns:
            Sketch: The converted sketch object.

        Raises:
            ValueError: If the input array is not of shape (N, 3).
        """
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError("FS-COCO npy must be (N,3)")

        penup = {1, 2}
        points: list[tuple[float, float]] = []
        curves: list[Curve] = []

        def _append_curve(p0: tuple[float, float], p1: tuple[float, float]):
            line = np.array([p0, p1], dtype=float)  # (2,2)
            cubic = line_to_cubic(line)  # Returns a Curve object
            curves.append(cubic)

        for x, y, f in data:
            points.append((float(x), float(y)))
            if int(f) in penup:
                if len(points) > 1:
                    for i in range(len(points) - 1):
                        _append_curve(points[i], points[i + 1])
                points = []

        if len(points) > 1:
            for i in range(len(points) - 1):
                _append_curve(points[i], points[i + 1])

        return Sketch(paths=[SKPath(curves=curves)]) if curves else Sketch(paths=[])

    def extra_repr(self) -> str:
        """
        Returns a string representation of the dataset for debugging.

        Returns:
            str: A string summarizing the dataset.
        """
        total = len(self.items_metadata) if hasattr(self, "items_metadata") else 0
        shards = sorted(self.items_metadata["shard"].unique().tolist()) if total else []
        return f"Files: {total} | Shards: {len(shards)} (1..100 expected)"


if __name__ == "__main__":
    import psutil
    from rich.console import Console
    import time
    import tempfile

    console = Console()
    console.print("FSCOCO Dataset Test", style="bold blue")

    # 1) Default download (CISLAB) to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        console.print("[green]1) Load from CISLAB CDN (if available)[/green]")
        ds = FSCOCO(cislab_source=True, load_all=True)
        console.print(ds)
        mem = psutil.Process().memory_info().rss / 1024 / 1024
        console.print(f"Memory: {mem:.2f} MB")
        n = min(50, len(ds))
        t0 = time.time()
        _ = [ds[i] for i in range(n)]
        console.print(f"Loaded {n} samples in {time.time() - t0:.3f}s")

    # 2) Test preloading all items at once
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     console.print("[green]2) Load All[/green]")
    #     ds = FSCOCO(root=tmpdir, cislab_source=False, load_all=True)
    #     console.print(ds)
    #     mem = psutil.Process().memory_info().rss / 1024 / 1024
    #     console.print(f"Memory: {mem:.2f} MB")
    #     n = min(100, len(ds))
    #     t0 = time.time()
    #     _ = [ds[i] for i in range(n)]
    #     console.print(f"Loaded {n} samples in {time.time() - t0:.3f}s")
