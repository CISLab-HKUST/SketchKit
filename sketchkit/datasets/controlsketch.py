import os
import shutil

import glob
import numpy as np
import gdown

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path as PathLibPath

from svg.path import parse_path, path
from sketchkit.utils.dataset import SketchDataset
from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.transform import line_to_cubic
from sketchkit.utils.file import extract_files, download_with_wget, dir_md5, CISLAB_CDN


_INVALID = {"rect", "circle", "ellipse", "line", "polyline", "polygon"}


def parse_svg(svg_path: str) -> list[Path]:
    """Parse an SVG file into a list of Path objects.

    Args:
        svg_path (str): Path to the SVG file to parse.

    Returns:
        list[Path]: Parsed sketch represented as a list of Path objects.

    Raises:
        AssertionError: If a transform attribute is encountered.
        Exception: If an arc segment is encountered.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    paths: list[Path] = []

    for elem in root.iter():
        tag = elem.tag.split("}", 1)[-1]
        if tag in _INVALID:
            continue
        if tag != "path":
            continue

        if "transform" in elem.attrib:
            raise AssertionError("Transform not supported")

        d = elem.attrib.get("d", "")
        if not d:
            continue

        cps: list[tuple[float, float]] = []
        ps = parse_path(d)
        for seg in ps:
            t = type(seg)

            if t == path.Move:
                cps.append((seg.start.real, seg.start.imag))

            elif t == path.CubicBezier:
                if not cps:
                    s = seg.start
                    cps.append((s.real, s.imag))
                c1, c2, e = seg.control1, seg.control2, seg.end
                cps.extend([(c1.real, c1.imag), (c2.real, c2.imag), (e.real, e.imag)])

            elif t == path.Line:
                if not cps:
                    s = seg.start
                    cps.append((s.real, s.imag))
                s, e = seg.start, seg.end
                p0 = (s.real, s.imag)
                p1 = (e.real, e.imag)
                cubic = line_to_cubic(np.stack([p0, p1], axis=0))
                cps.extend(
                    [
                        (cubic[1, 0], cubic[1, 1]),
                        (cubic[2, 0], cubic[2, 1]),
                        (cubic[3, 0], cubic[3, 1]),
                    ]
                )

            elif t == path.Arc:
                raise Exception("Arc not supported")

        if len(cps) > 1 and (len(cps) - 1) % 3 == 0:
            n = (len(cps) - 1) // 3
            curves: List[Curve] = []
            for i in range(n):
                p0 = cps[i * 3 + 0]
                c1 = cps[i * 3 + 1]
                c2 = cps[i * 3 + 2]
                p3 = cps[i * 3 + 3]
                curves.append(
                    Curve(
                        Vertex(float(p0[0]), float(p0[1])),
                        Vertex(float(p3[0]), float(p3[1])),
                        Point(float(c1[0]), float(c1[1])),
                        Point(float(c2[0]), float(c2[1])),
                    )
                )
            if curves:
                paths.append(Path(curves=curves))

    return paths


class ControlSketch(SketchDataset):
    """ControlSketch dataset loader (SketchDataset-style).

    Directory structure (after download & extract):
        controlsketch_sketches/
          ├── train/<category>/*.svg
          ├── validation/<category>/*.svg
          └── test/<category>/*.svg

    References:
        - Original dataset: https://swiftsketch.github.io/#controlsketch-data
        - Paper: "SwiftSketch: A Diffusion Model for Image-to-Vector Sketch Generation" by E Arar, 2025
    """

    md5_sum = "c45dad0c08988df3d4036e85e5363e8a"
    metadata = ["id", "sub_id", "category", "split", "filepath"]

    def __init__(
        self,
        root: str | PathLibPath | None = None,
        load_all: bool = False,
        cislab_source: bool = False,
    ):
        """Initialize the ControlSketch dataset.

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
        ds_root = os.path.join(self.root, "controlsketch_sketches")
        if not os.path.isdir(ds_root):
            return False
        if getattr(self, "md5_sum", None) in (None, "", 0):
            return True
        return dir_md5(ds_root) == self.md5_sum

    def _download(self):
        """Download and extract the ControlSketch dataset.

        Downloads the dataset archive from CISLAB CDN or Google Drive,
        extracts it into the dataset root, and removes the source file if successful.
        """
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

        zip_name = "controlsketch_sketches.zip"
        zip_path = os.path.join(self.root, zip_name)

        if self.cislab_source:
            try:
                url = f"{CISLAB_CDN}/datasets/{self.__class__.__name__}/{zip_name}"
                print(f"Downloading ControlSketch from CISLAB CDN: {url}")
                download_with_wget(url, file_path=zip_path, pg_bar=True)
            except Exception:
                print(
                    "CISLAB mirror unavailable. Falling back to Google Drive (gdown)."
                )
                file_id = "1-kffChm6RXwTuP8qxUOT-QDf8-bSCH0w"
                if not os.path.exists(zip_path):
                    print("Downloading ControlSketch from Google Drive…")
                    gdown.download(id=file_id, output=zip_path, quiet=False)
                    print("Download finished.")
        else:
            file_id = "1-kffChm6RXwTuP8qxUOT-QDf8-bSCH0w"
            if not os.path.exists(zip_path):
                print("Downloading ControlSketch from Google Drive…")
                gdown.download(id=file_id, output=zip_path, quiet=False)
                print("Download finished.")

        try:
            print("Extracting ControlSketch…")
            extract_files(zip_path, self.root, remove_sourcefile=True)
            print("Extraction finished.")
        except Exception as e:
            raise e
        
        print("MD5 Checksum:", dir_md5(self.root))

    def _load_items_metadata(self):
        """Build metadata index for dataset items.

        Scans the extracted dataset directory for train/validation/test splits,
        collects all SVG files with their category and split, and saves results
        into a cached Parquet file.

        Raises:
            FileNotFoundError: If the dataset directory is not found.
        """
        ds_root = os.path.join(self.root, "controlsketch_sketches")
        if not os.path.isdir(ds_root):
            raise FileNotFoundError(f"Not found dataset root: {ds_root}")

        meta_cache = os.path.join(self.root, ".metadata.parquet")

        if os.path.exists(meta_cache):
            try:
                items = pd.read_parquet(meta_cache)
                if set(items.columns.tolist()) == set(self.metadata):
                    self.items_metadata = items
                    self.raw_data = [None] * len(self.items_metadata)
                    return
            except Exception:
                pass
            try:
                os.remove(meta_cache)
            except Exception:
                pass

        splits = ["train", "validation", "test"]
        rows = []
        gid = 0

        for sp in splits:
            sp_dir = os.path.join(ds_root, sp)
            if not os.path.isdir(sp_dir):
                continue

            cats = [
                d for d in os.listdir(sp_dir) if os.path.isdir(os.path.join(sp_dir, d))
            ]
            for cat in sorted(cats):
                cdir = os.path.join(sp_dir, cat)
                svgs = sorted(glob.glob(os.path.join(cdir, "*.svg")))
                for sub_id, fp in enumerate(svgs):
                    row = {
                        "id": gid,
                        "sub_id": sub_id,
                        "category": cat,
                        "split": sp,
                    }
                    if "filepath" in self.metadata:
                        row["filepath"] = fp
                    rows.append(row)
                    gid += 1

        self.items_metadata = pd.DataFrame(rows, columns=self.metadata)

        try:
            self.items_metadata.to_parquet(meta_cache, compression="zstd")
        except Exception:
            self.items_metadata.to_parquet(meta_cache)

        self.raw_data = [None] * len(self.items_metadata)

    def _load_all(self):
        """Load all sketch data into memory if load_all is enabled.

        Concatenates all sketch data from all categories and splits into a single
        numpy array for faster access. Only loads if self.load_all is True.
        """
        for idx in range(len(self.items_metadata)):
            fp = self.items_metadata.iloc[idx]["filepath"]
            self.raw_data[idx] = parse_svg(fp)

    def __getitem__(self, idx: int) -> Sketch:
        """Get one sketch by index.

        Args:
            idx (int): Global index of the sketch.

        Returns:
            Sketch: A Sketch object containing parsed paths.

        Raises:
            IndexError: If the index is out of range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        if self.raw_data[idx] is None:
            fp = self.items_metadata.iloc[idx]["filepath"]
            self.raw_data[idx] = parse_svg(fp)

        paths = self.raw_data[idx]
        return Sketch(paths=paths)

    def extra_repr(self) -> str:
        """Return dataset summary.

        Returns:
            str: A string with number of categories and sample counts
                for train/validation/test splits.
        """
        df = self.items_metadata

        def _n(sp):
            return int((df["split"] == sp).sum()) if isinstance(df, pd.DataFrame) else 0

        num_train = _n("train")
        num_val = _n("validation")
        num_test = _n("test")

        n_cat = df["category"].nunique() if len(df) else 0
        return f"Categories: {n_cat}\nTrain: {num_train} Validation: {num_val} Test samples: {num_test}"


if __name__ == "__main__":
    import psutil
    from rich.console import Console
    import time

    console = Console()

    console.print("ControlSketch Dataset Test", style="bold blue")
    console.print()

    # console.print("1. Default Load", style="green")
    # dataset = ControlSketch()
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
    # dataset = ControlSketch(load_all=True)
    # # show brief information of the dataset
    # console.print(dataset)
    # process = psutil.Process()
    # memory_mb = process.memory_info().rss / 1024 / 1024
    # console.print(f"Current memory usage: {memory_mb:.2f} MB")
    # # search data with "category = cow" and "split = test"
    # cows = dataset.items_metadata[
    #     (dataset.items_metadata["category"] == "cow")
    #     & (dataset.items_metadata["split"] == "test")
    # ]
    # start_time = time.time()
    # cow_sketch = [dataset[row.id] for _, row in cows[:100].iterrows()]
    # console.print(f"Loading 100 sketches in {time.time() - start_time}")
    # del dataset, cows

    # console.print()

    console.print("3. Load from CISLAB CDN", style="green")

    # tempfile can create a temporary directory
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = ControlSketch(root=tmpdir, cislab_source=True)
        # show brief information of the dataset
        console.print(dataset)
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        console.print(f"Current memory usage: {memory_mb:.2f} MB")
        bears = dataset.items_metadata[
            (dataset.items_metadata["category"] == "bear")
            & (dataset.items_metadata["split"] == "validation")
        ]
        start_time = time.time()
        bears_sketch = [dataset[row.id] for _, row in bears[:100].iterrows()]
        console.print(f"Loading 100 sketches in {time.time() - start_time}")
        del dataset, bears

        console.print()

        from PIL import Image
        from sketchkit.renderer.cairo_renderer import CairoRenderer

        renderer = CairoRenderer(256, (1, 1, 1))
        raster_image = renderer.render(bears_sketch[0])
        outpath = "test.png"
        if outpath is not None:
            raster_image_png = Image.fromarray(raster_image, "RGB")
            raster_image_png.save(outpath, "PNG")

