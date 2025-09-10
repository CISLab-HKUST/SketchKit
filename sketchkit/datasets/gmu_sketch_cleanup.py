import os
import shutil
import ssl

import glob
import xml.etree.ElementTree as ET
from pathlib import Path as PathLibPath
import numpy as np
import pandas as pd
from svg.path import parse_path, Path as SvgPath, Move, CubicBezier, Line, Arc

from sketchkit.core.sketch import Curve, Vertex, Point, Sketch, Path
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import download_with_wget, dir_md5, CISLAB_CDN, extract_files

INVALID_SVG_SHAPES = {"rect", "circle", "ellipse", "line", "polyline", "polygon"}


def parse_gmu_svg(svg_file: str) -> tuple[tuple[int, int], list[list[np.ndarray]], int]:
    """Parses a GMU SVG file into (width, height), path_list, and total_segment_num.

    Args:
        svg_file: Path to the SVG file.

    Returns:
        A tuple containing:
        - (width, height): Dimensions of the SVG canvas.
        - path_list: A list of paths, where each path contains multiple cubic segments.
        - total_segment_num: Total number of cubic segments in the SVG.
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()

    view_box = root.get("viewBox")
    if view_box:
        view_x, view_y, view_w, view_h = [float(x) for x in view_box.split()]
        width = int(view_w - view_x)
        height = int(view_h - view_y)
    else:

        def _to_int_px(v: str | None) -> int:
            if v is None:
                return 0
            v = v.replace("px", "")
            try:
                return int(float(v))
            except Exception:
                return 0

        width = _to_int_px(root.get("width"))
        height = _to_int_px(root.get("height"))

    path_list: list[list[np.ndarray]] = []
    total_segment_num = 0

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag in INVALID_SVG_SHAPES:
            continue
        if tag == "path":
            if "transform" in elem.attrib:
                raise Exception(f"Transform not supported in {svg_file}")
            d = elem.attrib.get("d")
            if not d:
                continue
            ps: SvgPath = parse_path(d)
            cps: list[tuple[float, float]] = (
                []
            )  # Records (x, y) in order: start1 + (c1, c2, pt)*N
            for seg in ps:
                if isinstance(seg, Move):
                    # End the previous path
                    if len(cps) > 1:
                        seg_num = (len(cps) - 1) // 3
                        strokes = [
                            np.array(cps[i * 3 : i * 3 + 4], dtype=float)
                            for i in range(seg_num)
                        ]
                        path_list.append(strokes)
                        total_segment_num += seg_num
                    cps = [(seg.start.real, seg.start.imag)]
                elif isinstance(seg, CubicBezier):
                    s, c1, c2, e = seg.start, seg.control1, seg.control2, seg.end
                    if not cps:
                        cps.append((s.real, s.imag))
                    cps.append((c1.real, c1.imag))
                    cps.append((c2.real, c2.imag))
                    cps.append((e.real, e.imag))
                elif isinstance(seg, Line):
                    s, e = seg.start, seg.end
                    sx, sy, ex, ey = s.real, s.imag, e.real, e.imag
                    if not cps:
                        cps.append((sx, sy))
                    # Convert the line to an equivalent cubic Bezier curve
                    c1x = 2.0 / 3.0 * sx + 1.0 / 3.0 * ex
                    c1y = 2.0 / 3.0 * sy + 1.0 / 3.0 * ey
                    c2x = 1.0 / 3.0 * sx + 2.0 / 3.0 * ex
                    c2y = 1.0 / 3.0 * sy + 2.0 / 3.0 * ey
                    cps.append((c1x, c1y))
                    cps.append((c2x, c2y))
                    cps.append((ex, ey))
                elif isinstance(seg, Arc):
                    raise Exception("Arc not supported")

            # Finalize the last path
            if len(cps) > 1:
                seg_num = (len(cps) - 1) // 3
                strokes = [
                    np.array(cps[i * 3 : i * 3 + 4], dtype=float)
                    for i in range(seg_num)
                ]
                path_list.append(strokes)
                total_segment_num += seg_num

    return (width, height), path_list, total_segment_num


class GMUSketchCleanup(SketchDataset):
    """GMU Rough Sketch Cleanup dataset (SVG parsing version).

    Directory structure (after extraction):
    Benchmark_Dataset/
      ├── GT/
      ├── Rough/JPG/
      ├── Rough/PNG/
      ├── Rough/SVG/
      └── sketch_tags.csv

    - Only the `id` column is mandatory; additional columns like `split` (GT/Rough)
      and `file_path` (SVG path) are optional.
    - __getitem__: Reads from raw_data; if missing, parses the SVG from disk and caches it.
    - Supports downloading from the official website or CISLAB mirror.
    """

    # Replace this value with the actual directory MD5 (MD5 of the Benchmark_Dataset directory or the entire root after download and extraction).
    md5_sum: str = "8dbea6e9cd42810c80f41595a633ff33"

    # Only `id` is mandatory; add columns as needed.
    metadata = ["id", "split", "file_path"]

    def __init__(
        self,
        root: str | PathLibPath | None = None,
        load_all: bool = False,
        cislab_source: bool = False,
    ):
        super().__init__(root, load_all=load_all, cislab_source=cislab_source)

    def _check_integrity(self) -> bool:
        """Checks the integrity of the cached dataset by comparing MD5 checksums."""
        print(f"Checking integrity of cached {self.__class__.__name__} dataset...")
        current_md5 = dir_md5(self.root)
        return current_md5 == self.md5_sum

    def _download(self):
        """Downloads and extracts the GMU-Sketch-Cleanup dataset.

        - Official: https://cragl.cs.gmu.edu/sketchbench/Benchmark_Dataset.zip
        - CISLAB: {CISLAB_CDN}/datasets/GMUSketchCleanup/Benchmark_Dataset.zip
        """
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

        if self.cislab_source:
            url = (
                f"{CISLAB_CDN}/datasets/{self.__class__.__name__}/Benchmark_Dataset.zip"
            )
        else:
            ssl._create_default_https_context = ssl._create_unverified_context
            url = "https://cragl.cs.gmu.edu/sketchbench/Benchmark_Dataset.zip"

        zip_fp = os.path.join(self.root, "Benchmark_Dataset.zip")

        download_with_wget(url, file_path=zip_fp, desc="Downloading GMU-Sketch-Cleanup")

        extract_files(zip_fp, self.root)
        print("MD5 Checksum:", dir_md5(self.root))

    def _load_items_metadata(self):
        """Loads or generates metadata for the dataset."""
        meta_path = os.path.join(self.root, ".metadata.parquet")
        bench_root = os.path.join(self.root, "Benchmark_Dataset")

        if not os.path.exists(meta_path):
            rows = []

            gt_svg = sorted(
                glob.glob(os.path.join(bench_root, "GT", "**", "*.svg"), recursive=True)
            )
            for fp in gt_svg:
                rows.append({"split": "GT", "file_path": fp})

            rough_svg = sorted(
                glob.glob(
                    os.path.join(bench_root, "Rough", "SVG", "**", "*.svg"),
                    recursive=True,
                )
            )
            for fp in rough_svg:
                rows.append({"split": "Rough", "file_path": fp})

            # Generate DataFrame: only `id` is mandatory
            items = pd.DataFrame(rows, columns=["split", "file_path"])
            items.insert(0, "id", range(len(items)))  # Global id

            # Retain only declared columns
            self.items_metadata = items[self.metadata].copy()
            self.items_metadata.to_parquet(meta_path, compression="zstd")
        else:
            items_metadata = pd.read_parquet(meta_path)
            if set(items_metadata.columns.tolist()) != set(self.metadata):
                os.remove(meta_path)
                return self._load_items_metadata()
            self.items_metadata = items_metadata

        self.raw_data = [None] * len(self.items_metadata)

    def _load_all(self):
        """Preloads all items into memory."""
        for i in range(len(self.items_metadata)):
            self.raw_data[i] = self._load_one(i)

    def __getitem__(self, idx: int) -> Sketch:
        """Retrieves a single item by index.

        Args:
            idx: Index of the item.

        Returns:
            A Sketch object.

        Raises:
            IndexError: If the index is out of range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        if self.raw_data[idx] is None:
            # (Optional) You can also perform "batch loading by split" here to reduce I/O operations.
            self.raw_data[idx] = self._load_one(idx)

        (w, h), path_list, _ = self.raw_data[idx]
        paths = []
        for p in path_list:
            curves = []
            for seg in p:
                # seg: (4,2) -> p0, c1, c2, p3
                p0 = Vertex(seg[0][0], seg[0][1])
                c1 = Point(seg[1][0], seg[1][1])
                c2 = Point(seg[2][0], seg[2][1])
                p3 = Vertex(seg[3][0], seg[3][1])
                curves.append(Curve(p0, p3, c1, c2))
            paths.append(Path(curves=curves))
        return Sketch(paths=paths)

    def extra_repr(self) -> str:
        """Returns additional string representation of the dataset."""
        n = len(self.items_metadata)
        splits = (
            self.items_metadata["split"].value_counts().to_dict()
            if "split" in self.items_metadata.columns
            else {}
        )
        parts = [f"Items: {n}"]
        if splits:
            parts.append("Splits: " + ", ".join(f"{k}={v}" for k, v in splits.items()))
        return " | ".join(parts)

    def _load_one(self, idx: int):
        """Loads a single item from disk.

        Args:
            idx: Index of the item.

        Returns:
            Parsed SVG data.
        """
        row = self.items_metadata.iloc[idx]
        fp = row["file_path"]
        if not fp.lower().endswith(".svg"):
            raise TypeError("Current item is not SVG; cannot construct Sketch")
        return parse_gmu_svg(fp)


if __name__ == "__main__":
    import psutil
    from rich.console import Console
    import time
    import tempfile

    console = Console()
    console.print("GMUSketchCleanup Dataset Test", style="bold blue")

    # 1) Default download (CISLAB) to a temporary directory
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     console.print("[green]1) Load from CISLAB CDN (if available)[/green]")
    #     ds = GMUSketchCleanup(root=tmpdir, cislab_source=True, load_all=False)
    #     console.print(ds)
    #     mem = psutil.Process().memory_info().rss / 1024 / 1024
    #     console.print(f"Memory: {mem:.2f} MB")
    #     n = min(50, len(ds))
    #     t0 = time.time()
    #     _ = [ds[i] for i in range(n)]
    #     console.print(f"Loaded {n} samples in {time.time() - t0:.3f}s")

    # 2) Test preloading all items at once
    with tempfile.TemporaryDirectory() as tmpdir:
        console.print("[green]2) Load All[/green]")
        ds = GMUSketchCleanup(root=tmpdir, cislab_source=True, load_all=True)
        console.print(ds)
        mem = psutil.Process().memory_info().rss / 1024 / 1024
        console.print(f"Memory: {mem:.2f} MB")
        n = min(100, len(ds))
        t0 = time.time()
        _ = [ds[i] for i in range(n)]
        console.print(f"Loaded {n} samples in {time.time() - t0:.3f}s")
