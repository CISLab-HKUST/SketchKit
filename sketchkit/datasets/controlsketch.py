import os
import glob
import numpy as np
import gdown
import xml.etree.ElementTree as ET
from typing import List

from svg.path import parse_path, path
from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.general_utils import extract_files, line_to_cubic


def download_controlsketch(output_folder: str):
    """Download and extract the ControlSketch dataset.

    This function downloads the ControlSketch dataset archive from Google Drive
    and extracts it into the specified output folder. The final structure is:

        <output_folder>/controlsketch_sketches/
          ├── train/<category>/*.svg
          ├── validation/<category>/*.svg
          └── test/<category>/*.svg

    Args:
        output_folder (str): Directory where the dataset will be stored.
    """
    os.makedirs(output_folder, exist_ok=True)

    ds_root = os.path.join(output_folder, "controlsketch_sketches")
    if os.path.isdir(ds_root):
        print("ControlSketch already exists. Skip download and extract.")
        return

    file_id = "1-kffChm6RXwTuP8qxUOT-QDf8-bSCH0w"
    zip_name = "controlsketch_sketches.zip"
    zip_path = os.path.join(output_folder, zip_name)

    if not os.path.exists(zip_path):
        print("Downloading ControlSketch...")
        gdown.download(id=file_id, output=zip_path, quiet=False)
        print("Download finished.")

    print("Extracting ControlSketch...")
    extract_files(zip_path, output_folder, remove_sourcefile=True)
    print("Extraction finished.")


_INVALID = {"rect", "circle", "ellipse", "line", "polyline", "polygon"}

def parse_svg(svg_path: str) -> List[Path]:
    """Parse an SVG file into a list of Path objects.

    Each '<path>' element in the SVG becomes a 'Path' object containing
    one or more cubic Bézier curves. Line segments are promoted to cubic
    curves via 'line_to_cubic'. Transforms and arcs are not supported.

    Args:
        svg_path (str): Path to the SVG file to parse.

    Returns:
        List[Path]: A list of 'Path' objects extracted from the SVG file.

    Raises:
        AssertionError: If a 'transform' attribute is encountered.
        Exception: If an arc segment is encountered.
    """
    tree = ET.parse(svg_path)
    root = tree.getroot()

    paths: List[Path] = []

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
                cps.extend([(cubic[1, 0], cubic[1, 1]),
                            (cubic[2, 0], cubic[2, 1]),
                            (cubic[3, 0], cubic[3, 1])])

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


class ControlSketch:
    """ControlSketch dataset loader.

    The ControlSketch dataset consists of vector sketches stored as SVG files.
    It is organized into splits ('train', 'validation', 'test') and categories.

    Expected directory structure:
        controlsketch_sketches/
          ├── train/<category>/*.svg
          ├── validation/<category>/*.svg
          └── test/<category>/*.svg

    Attributes:
        category (str | None): Specific category to load, or None for all.
        split (str): Dataset split, one of {"train", "validation", "test", "all"}.
        filepaths (list[str]): List of filepaths to the SVGs.

    Example:
        >>> ds = ControlSketch(category="ant", split="train")
        >>> print(len(ds))
        >>> sketch = ds[0]
        >>> print(sketch.curve_num, sketch.path_num)
    """

    def __init__(self, category=None, split="train", data_base=".cache/sketchkit/datasets"):
        """Initialize the ControlSketch dataset.

        Args:
            category (str | None, optional): Category name (e.g., "ant").
                If None, all categories are included. Defaults to None.
            split (str, optional): Dataset split ("train", "validation", "test", "all").
                Defaults to "train".
            data_base (str, optional): Relative path under $HOME for dataset storage.
                Defaults to ".cache/sketchkit/datasets".

        Raises:
            FileNotFoundError: If the dataset root directory is missing.
            ValueError: If the given split is invalid.
            RuntimeError: If no SVG files are found under the split/category.
        """
        self.category = category
        self.split = split

        data_root = os.path.join(os.environ.get("HOME"), data_base)
        os.makedirs(data_root, exist_ok=True)

        download_controlsketch(data_root)

        ds_root = os.path.join(data_root, "controlsketch_sketches")
        if not os.path.isdir(ds_root):
            raise FileNotFoundError(f"Not found: {ds_root}")

        if split not in ["train", "validation", "test", "all"]:
            raise ValueError(f"Unknown split: {split}")

        splits = ["train", "validation", "test"] if split == "all" else [split]
        filepaths: list[str] = []
        for sp in splits:
            sp_dir = os.path.join(ds_root, sp)
            if not os.path.isdir(sp_dir):
                continue

            cats = (
                [self.category]
                if (self.category and os.path.isdir(os.path.join(sp_dir, self.category)))
                else [d for d in os.listdir(sp_dir) if os.path.isdir(os.path.join(sp_dir, d))]
            )
            for cat in cats:
                cdir = os.path.join(sp_dir, cat)
                if not os.path.isdir(cdir):
                    continue
                filepaths.extend(glob.glob(os.path.join(cdir, "*.svg")))

        self.filepaths = sorted(filepaths)
        if not self.filepaths:
            raise RuntimeError("No SVG found under the given split/category.")

    def __len__(self) -> int:
        """Return the number of sketches in the dataset."""
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Sketch:
        """Get a sketch by index.

        Args:
            idx (int): Index of the sketch to load.

        Returns:
            Sketch: A 'Sketch' object containing parsed paths and curves.
        """
        svg_path = self.filepaths[idx]
        paths = parse_svg(svg_path)
        return Sketch(paths=paths)
