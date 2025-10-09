import os
import shutil
from pathlib import Path as PathLibPath

import numpy as np
import pandas as pd
from tqdm import tqdm

from sketchkit.core.sketch import Sketch, Path, Curve
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import download_with_wget, dir_md5, CISLAB_CDN, extract_files
from sketchkit.utils.transform import line_to_cubic


class FSCOCO(SketchDataset):
    """FS-COCO dataset loader for vector sketches stored in `.npy` format.

    This dataset class can download, verify, index, and load FS-COCO vector
    sketches. Each sketch is represented as polyline points and converted to
    cubic Bézier curves.

    Attributes:
        md5_sum (str): MD5 checksum for verifying dataset integrity.
        metadata (list[str]): Column names used in the metadata table.
    """

    md5_sum: str = "5e5c08c2e6877b164b5d04f3ce8b9f89"
    metadata = ["id", "shard", "file_path"]

    def __init__(
            self,
            root: str | PathLibPath | None = None,
            load_all: bool = False,
            cislab_source: bool = False,
    ):
        """Initialize the FSCOCO dataset.

        Args:
            root: Root directory of the dataset. The dataset will be extracted
                under `<root>/fscoco/`. If None, the parent class default is used.
            load_all: If True, preload all `.npy` files into memory.
            cislab_source: If True, download from the CISLAB mirror; otherwise
                use the official source.
        """
        super().__init__(root, load_all=load_all, cislab_source=cislab_source)

    def _check_integrity(self) -> bool:
        """Check on-disk dataset integrity.

        Returns:
            bool: True if the dataset directory exists and its MD5 matches
            the expected checksum; False otherwise.
        """
        print(f"Checking integrity of cached {self.__class__.__name__} dataset...")
        root = PathLibPath(self.root)
        data_dir = root / "fscoco" / "vector_sketches"
        if not data_dir.exists():
            return False
        current_md5 = dir_md5(root / "fscoco")
        return current_md5 == self.md5_sum

    def _download(self):
        """Download and extract FS-COCO if the integrity check fails.

        The archive is fetched from the CISLAB mirror or the official host,
        extracted into `self.root`, and a checksum is printed for verification.
        """
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

        url = (
            f"{CISLAB_CDN}/datasets/FSCOCO/fscoco.tar.gz"
            if self.cislab_source
            else "http://cvssp.org/data/fscoco/fscoco.tar.gz"
        )

        print(f"[FS-COCO] Downloading from: {url}")

        archive = os.path.join(self.root, "fscoco.tar.gz")
        download_with_wget(url, file_path=archive, desc="Downloading FS-COCO")

        print("[FS-COCO] Extracting...")
        extract_files(archive, self.root)

        print("MD5 Checksum of FS-COCO:", dir_md5(self.root))

    def _load_items_metadata(self):
        """Build or load the items metadata table.

        Scans the dataset directory to create a table with the following columns:
        - `id`: A globally unique integer identifier for each sketch.
        - `shard`: The shard index (1..100).
        - `file_path`: Relative path to the `.npy` file (relative to `self.root`).

        The table is cached in `.metadata.parquet` for faster subsequent loads.
        """
        meta_path = PathLibPath(self.root) / ".metadata.parquet"
        data_dir = PathLibPath(self.root) / "fscoco" / "vector_sketches"

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
                        "file_path": str(npy_file.relative_to(self.root)),
                    }
                )
                idx += 1

        df = pd.DataFrame(rows, columns=self.metadata)
        self.items_metadata = df
        self.items_metadata.to_parquet(meta_path, compression="zstd")
        self.raw_data = [None] * len(self.items_metadata)

    def _load_all(self):
        """Preload all `.npy` arrays into memory.

        This is useful when `load_all=True` is requested and you want to avoid
        disk I/O during iteration.
        """
        for i in tqdm(range(len(self.items_metadata)), desc="Loading all FS-COCO"):
            self.raw_data[i] = self._load_single_array(i)

    def _get_single(self, idx: int) -> Sketch:
        """Get a single sketch by index.

        Args:
            idx: Index of the sketch to retrieve.

        Returns:
            Sketch: A `Sketch` object built from the underlying array.

        Raises:
            IndexError: If `idx` is out of bounds.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        if self.raw_data[idx] is None:
            self.raw_data[idx] = self._load_single_array(idx)
        return self._array_to_sketch(self.raw_data[idx])

    def _load_single_array(self, idx: int) -> np.ndarray:
        """Load a single `.npy` array for the given index.

        Args:
            idx: Row index in the metadata table.

        Returns:
            np.ndarray: Array of shape `(N, 3)` containing `[x, y, flag]`.
        """
        rel = self.items_metadata.iloc[idx]["file_path"]
        path = PathLibPath(self.root) / rel
        return np.load(str(path))

    @staticmethod
    def _array_to_sketch(data: np.ndarray) -> Sketch:
        """Convert a `(N, 3)` array to a `Sketch` with multiple `Path`s.

        Each continuous stroke (sequence of points until a pen-up/end flag) is
        converted into one `Path`. Adjacent point pairs are mapped to cubic
        Bézier `Curve`s using `line_to_cubic`.

        Args:
            data: Array of shape `(N, 3)` with columns `[x, y, flag]`. Flag
                meanings (common convention): `0`=pen-down/continue, `1`=pen-up,
                `2`=end-of-drawing.

        Returns:
            Sketch: A sketch composed of zero or more paths.

        Raises:
            ValueError: If `data` is not a 2D array with exactly 3 columns.
        """
        # Validate shape early to produce clear errors.
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError("FS-COCO npy must be (N,3)")

        penup = {1, 2}
        paths: list[Path] = []
        points: list[tuple[float, float]] = []

        def flush_current_stroke(pts: list[tuple[float, float]]):
            """Convert the current stroke points into a `Path` and append it.

            Args:
                pts: The accumulated points (x, y) belonging to one stroke.
            """
            if len(pts) <= 1:
                return
            curves: list[Curve] = []
            for i in range(len(pts) - 1):
                # Map each polyline edge to a cubic Bézier curve.
                line = np.array([pts[i], pts[i + 1]], dtype=float)
                cubic = line_to_cubic(line)
                curves.append(cubic)
            if curves:
                paths.append(Path(curves=curves))

        # Accumulate points for a stroke; on pen-up/end, finalize the current path.
        for x, y, f in data:
            points.append((float(x), float(y)))
            f = int(f)

            if f == 0:
                continue

            if f == 1:  # pen-up: end current stroke and continue parsing
                flush_current_stroke(points)
                points = []
                continue

            if f == 2:  # end-of-drawing: end current stroke and stop parsing
                flush_current_stroke(points)
                points = []
                break

        # If the last stroke does not end with a pen-up flag, flush it here.
        flush_current_stroke(points)

        return Sketch(paths=paths)

    def extra_repr(self) -> str:
        """Return a short human-readable summary for debugging.

        Returns:
            str: Summary string containing file count and shard coverage.
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

    # 1) Default download (CISLAB) to a temporary directory.
    with tempfile.TemporaryDirectory() as tmpdir:
        console.print("[green]1) Load from CISLAB CDN (if available)[/green]")
        ds = FSCOCO(root=tmpdir, cislab_source=True, load_all=True)
        console.print(ds)
        mem = psutil.Process().memory_info().rss / 1024 / 1024
        console.print(f"Memory: {mem:.2f} MB")
        n = min(50, len(ds))
        t0 = time.time()
        _ = [ds[i] for i in range(n)]
        console.print(f"Loaded {n} samples in {time.time() - t0:.3f}s")

    # 2) Test preloading all items at once.
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
