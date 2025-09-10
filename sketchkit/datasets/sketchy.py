import os
import shutil
import traceback
import re

import xml.etree.ElementTree as ET
import pandas as pd
import gdown
from svg.path import parse_path, path

from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import download_with_wget, dir_md5, CISLAB_CDN, extract_files
from sketchkit.utils.transform import line_to_cubic


invalid_svg_shapes = ["rect", "circle", "ellipse", "line", "polyline", "polygon"]


def parse_single_path(path_str: str):
    """Parse a single SVG path 'd' string into cubic Bézier control points.

    The returned list follows SketchKit's cubic Bézier convention:
    [start] + N * [ctrl1, ctrl2, end] where N is the number of segments.

    Args:
        path_str: The 'd' attribute string from an SVG <path>.

    Returns:
        A list of (x, y) control points if the path contains drawable segments.
        Returns None if the path is a trivial Move with no drawable segment.

    Raises:
        Exception: If unsupported SVG commands (e.g., Arc) are encountered.
    """
    ps = parse_path(path_str)

    control_points_list = []
    for item_i, path_item in enumerate(ps):
        path_type = type(path_item)

        if path_type == path.Move:
            # Record the starting anchor of this subpath
            start = path_item.start
            start_x, start_y = start.real, start.imag
            control_points_list.append((start_x, start_y))

        elif path_type == path.CubicBezier:
            # Ensure we have a starting anchor recorded; if not, add it.
            if len(control_points_list) == 0:
                s = path_item.start
                control_points_list.append((s.real, s.imag))
            start, control1, control2, end = (
                path_item.start,
                path_item.control1,
                path_item.control2,
                path_item.end,
            )
            control_points_list.append((control1.real, control1.imag))
            control_points_list.append((control2.real, control2.imag))
            control_points_list.append((end.real, end.imag))

        elif path_type == path.Arc:
            raise Exception("Arc is not supported")

        elif path_type == path.Line:
            # Use line_to_cubic from transform.py for line segment conversion
            if len(control_points_list) == 0:
                s = path_item.start
                control_points_list.append((s.real, s.imag))

            start, end = path_item.start, path_item.end
            line_points = [[start.real, start.imag], [end.real, end.imag]]

            # Convert line to cubic using transform.py function
            cubic_curve = line_to_cubic(line_points)
            control_points_list.append((cubic_curve.p_crtl1.x, cubic_curve.p_crtl1.y))
            control_points_list.append((cubic_curve.p_crtl2.x, cubic_curve.p_crtl2.y))
            control_points_list.append((cubic_curve.p_end.x, cubic_curve.p_end.y))

        else:
            raise Exception("Unknown path_type", path_type)

    assert (len(control_points_list) - 1) % 3 == 0

    if len(control_points_list) > 1:
        return control_points_list
    else:
        return None


def parse_svg(svg_file: str):
    """Parse an SVG file into SketchKit-compatible cubic Bézier curves.

    The parser assumes:
    * No transforms on <path> elements.
    * The root <svg> includes a 'viewBox' with x=y=0.

    Args:
        svg_file: Path to an SVG file.

    Returns:
        (sketch_width, sketch_height), path_list, total_segment_num
        where:
            - (sketch_width, sketch_height) is derived from the viewBox.
            - path_list is a list of curves: list of (N_curves, 4, 2).
            - total_segment_num is the total number of cubic segments.

    Raises:
        AssertionError: If unsupported shapes or transforms are encountered.
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # Robust viewBox / width/height parsing
    def _parse_number(val: str) -> float:
        # Strip units like 'px' or '%'; keep digits, minus, dot, and exponent markers
        sval = str(val).strip()
        sval = re.sub(r"[^0-9eE+\-\.]+", "", sval)
        return float(sval) if sval else 0.0

    view_box = root.get("viewBox") or root.get("viewbox")
    if view_box:
        parts = re.split(r"[\s,]+", view_box.strip())
        if len(parts) != 4:
            raise ValueError(f"SVG {svg_file} has malformed viewBox: '{view_box}'")
        view_x, view_y, view_width, view_height = map(_parse_number, parts)
    else:
        width = root.get("width")
        height = root.get("height")
        if width is None or height is None:
            raise ValueError(f"SVG {svg_file} missing viewBox and width/height")
        view_x, view_y = 0.0, 0.0
        view_width, view_height = _parse_number(width), _parse_number(height)
    assert int(view_x) == 0 and int(view_y) == 0
    sketch_width = int(view_width) - int(view_x)
    sketch_height = int(view_height) - int(view_y)

    path_list = []  # list of (N_curves, 4, 2)
    total_segment_num = 0

    for elem in root.iter():
        tag_suffix = elem.tag.split("}", 1)[-1]  # works with or without namespace

        # Skip unsupported primitive shapes quietly; we only consume <path> elements
        if tag_suffix in invalid_svg_shapes:
            continue

        if tag_suffix == "path":
            assert "transform" not in elem.attrib.keys()

            path_d = elem.attrib.get("d", None)
            if path_d is None:
                # Some malformed SVGs may include empty paths; skip safely.
                continue

            control_points_list = parse_single_path(path_d)  # (N, 2)
            if control_points_list is not None:
                segment_num = (len(control_points_list) - 1) // 3
                curve_list = []
                for i in range(segment_num):
                    curve = control_points_list[i * 3 : i * 3 + 4]  # (4, 2)
                    curve_list.append(curve)
                path_list.append(curve_list)
                total_segment_num += segment_num

    assert len(path_list) > 0
    return (sketch_width, sketch_height), path_list, total_segment_num


class Sketchy(SketchDataset):
    """The Sketchy Database (https://sketchy.eye.gatech.edu/), SVG subset.

    This loader expects the archive ``sketches-06-04.7z`` to extract into a
    ``sketchy/`` directory whose immediate subfolders are category names,
    each containing multiple ``.svg`` files. Any ``checked.txt`` or
    ``invalid.txt`` files present in category folders are ignored.

    After extraction, SVGs listed in per-class ``invalid.txt`` files are
    automatically deleted (this behavior can be disabled via
    ``prune_invalid=False`` in ``download_sketchy_database``).

    The data is represented using cubic Bézier curves consistent with SketchKit.

    Example:
        >>> data = Sketchy()
        >>> len(data)
        20000
        >>> s = data[0]
        >>> isinstance(s, Sketch)
        True

    Args:
        root: Root directory where the dataset will be stored.
        load_all: Whether to load all data into memory at initialization.
        prune: Whether to prune invalid SVGs during download.
    """

    md5_sum = "b0277b8d9413d7914971f1dae909f324"
    metadata = ["id", "pruned", "class_name", "filename", "path", "sketch_name"]
    gdrive_id: str = "1Qr8HhjRuGqgDONHigGszyHG_awCstivo"
    filename: str = "sketches-06-04.7z"

    def __init__(
        self,
        root: str | None = None,
        load_all: bool = False,
        cislab_source: bool = False,
    ):

        super().__init__(root=root, load_all=load_all, cislab_source=cislab_source)

    def _check_integrity(self) -> bool:
        """Check the integrity of the cached dataset using MD5 checksum.

        Returns:
            bool: True if the dataset integrity is verified, False otherwise.
        """
        print(f"Checking integrity of cached {self.__class__.__name__} dataset...")
        current_md5 = dir_md5(self.root)
        return current_md5 == self.md5_sum

    def _download(self):
        """Download and extract The Sketchy Database (SVG version)."""

        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

        print("Downloading Sketchy Database...")

        archive_path = os.path.join(self.root, self.filename)

        try:
            if self.cislab_source:
                cislab_url = f"{CISLAB_CDN}/datasets/Sketchy/{self.filename}"
                print(f"Attempting download from CISLAB: {cislab_url}")
                download_with_wget(
                    cislab_url, archive_path, desc="Downloading from CISLAB CDN"
                )
                print("CISLAB download completed successfully!")
            else:
                print("Using gdown for Google Drive download...")
                gdown.download(id=self.gdrive_id, output=archive_path, quiet=False)
                print("Google Drive download completed successfully!")
        except Exception:
            traceback.print_exc()
            print("Download failed.")
            return

        extract_files(archive_path, self.root, remove_sourcefile=True)
        print("MD5 Checksum:", dir_md5(self.root))

    def _load_items_metadata(self):
        """Load metadata for all items in the dataset into a pandas DataFrame."""
        self.src_dir = os.path.join(self.root, "sketches")
        if not os.path.exists(os.path.join(self.root, ".metadata.parquet")):
            class_names = [
                d
                for d in os.listdir(self.src_dir)
                # if os.path.isdir(os.path.join(self.root, d))
            ]
            class_names.sort()
            self.class_names = class_names

            metadata_list = []
            cnt = 0
            for class_name in self.class_names:
                class_base = os.path.join(self.src_dir, class_name)
                files = os.listdir(class_base)
                svg_files = sorted([f for f in files if f.lower().endswith(".svg")])

                invalid_files = []
                invalid_path = os.path.join(class_base, "invalid.txt")
                with open(invalid_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                for base in lines:
                    # Skip headings or section markers
                    if base.lower() in {
                        "invalid/error",
                        "unidentifiable/ambiguous",
                        "context",
                        "perspective",
                    }:
                        continue

                    invalid_files.append(os.path.join(class_name, f"{base}.svg"))

                for svg_file in svg_files:
                    sketch_name = f"{class_name}/{svg_file}"
                    metadata_list.append(
                        {
                            "id": cnt,
                            "pruned": sketch_name in invalid_files,
                            "class_name": class_name,
                            "filename": svg_file,
                            "path": os.path.join(class_base, svg_file),
                            "sketch_name": sketch_name,
                        }
                    )
                    cnt += 1

            if len(metadata_list) == 0:
                raise RuntimeError(
                    f"No SVG files found under '{self.svg_base}'. "
                    "Please ensure the correct archive is downloaded."
                )
            self.items_metadata = pd.DataFrame(metadata_list)
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

        self.raw_data = [None] * len(self.items_metadata)

    def _load_all(self):
        """Load all dataset items into memory."""
        for idx in range(len(self.items_metadata)):
            if self.raw_data[idx] is None:
                self.raw_data[idx] = self._load_sketch(idx)

    def _load_sketch(self, idx: int) -> Sketch:
        """Load a single sketch by index."""
        row = self.items_metadata.iloc[idx]
        svg_path = row["path"]

        # Parse original SVG into SketchVG-compatible parts
        image_sizes, path_list, total_segment_num = parse_svg(svg_path)

        paths = []
        for path_item in path_list:
            curves = []
            for raw_curve in path_item:
                p_start = Vertex(raw_curve[0][0], raw_curve[0][1])
                p_end = Vertex(raw_curve[3][0], raw_curve[3][1])
                p_ctrl1 = Point(raw_curve[1][0], raw_curve[1][1])
                p_ctrl2 = Point(raw_curve[2][0], raw_curve[2][1])
                curves.append(Curve(p_start, p_end, p_ctrl1, p_ctrl2))
            paths.append(Path(curves=curves))

        sketch = Sketch(paths=paths)
        return sketch

    def __getitem__(self, idx: int) -> Sketch:
        """Load a single sketch as a `Sketch` instance.

        Args:
            idx: Index into the dataset.

        Returns:
            A `Sketch` object composed of cubic Bézier `Path`/`Curve`s.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        if self.raw_data[idx] is None:
            self.raw_data[idx] = self._load_sketch(idx)
        return self.raw_data[idx]


if __name__ == "__main__":
    my_data = Sketchy()
    print(len(my_data))
    print(my_data[0])
    index = 10
    print(
        f"The file name of the {index}th sketch is: {my_data.items_metadata.iloc[index]['sketch_name']}"
    )

    print(my_data.items_metadata[my_data.items_metadata["pruned"] == True])
    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer

    renderer = CairoRenderer(512, (1, 1, 1))
    raster_image = renderer.render(my_data[index])
    outpath = f"test_{index}.png"
    if outpath is not None:
        raster_image_png = Image.fromarray(raster_image, "RGB")
        raster_image_png.save(outpath, "PNG")
