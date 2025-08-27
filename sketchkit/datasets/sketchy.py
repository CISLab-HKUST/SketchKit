import os
import sys
import traceback
import re
import requests
import xml.etree.ElementTree as ET

from svg.path import parse_path, path

from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.general_utils.md5sum import calculate_md5
from sketchkit.core import Sketch as SketchCore
import py7zr
import gdown


def _get_confirm_token(response) -> str | None:
    """Return Google Drive confirm token if the first request needs confirmation."""
    # Check special cookie first
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            return v
    # Fallback: parse HTML for confirm token
    try:
        text = response.text
    except Exception:
        return None
    m = re.search(r"confirm=([0-9A-Za-z_]+)&amp;", text)
    if not m:
        m = re.search(r"confirm=([0-9A-Za-z_]+)&", text)
    return m.group(1) if m else None


def _save_response_content(response, destination: str, chunk_size: int = 32768) -> None:
    """Stream HTTP response content to disk."""
    response.raise_for_status()
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)


def _download_from_gdrive(file_id: str, destination: str) -> None:
    """Download a file from Google Drive by file ID, handling confirm token."""
    URL = "https://drive.google.com/uc?export=download"
    with requests.Session() as session:
        # First request: may return a small HTML that requires confirmation
        response = session.get(
            URL, params={"id": file_id}, stream=True, allow_redirects=True
        )
        token = _get_confirm_token(response)
        if token:
            response.close()
            response = session.get(
                URL,
                params={"id": file_id, "confirm": token},
                stream=True,
                allow_redirects=True,
            )
        _save_response_content(response, destination)
    # Basic sanity check
    if not os.path.exists(destination) or os.path.getsize(destination) == 0:
        raise RuntimeError(
            "Downloaded file is empty; Google Drive may have blocked the request."
        )


def download_sketchy_database(
    output_folder: str,
    remove_sourcefile: bool = True,
    expected_md5: str | None = None,
    gdrive_id: str = "1Qr8HhjRuGqgDONHigGszyHG_awCstivo",
    filename: str = "sketches-06-04.7z",
    prune_invalid: bool = True,
):
    """Download and extract The Sketchy Database (SVG version).

    Downloads the 7z archive from Google Drive by file ID (preferred),
    verifies MD5 if provided, and extracts it into a ``sketches/`` directory.

    Args:
        output_folder: Directory where the archive will be saved and extracted.
        remove_sourcefile: If True, delete the downloaded 7z after extraction.
        expected_md5: Optional MD5 checksum string. If provided, the file's MD5
            will be verified and an AssertionError is raised on mismatch.
        gdrive_id: Google Drive file ID for the archive.
        filename: Local filename to save the downloaded 7z.

    Raises:
        AssertionError: If `expected_md5` is provided and the digest mismatches.
        FileNotFoundError: If extraction target is not created after extraction.
        RuntimeError: If extraction fails or download returns an empty file.
    """
    print("Downloading Sketchy Database...")

    os.makedirs(output_folder, exist_ok=True)
    archive_path = os.path.join(output_folder, filename)

    try:
        print("Using gdown for Google Drive download...")
        # Prefer gdown when available, it handles confirm tokens and cookies internally.
        gdown.download(id=gdrive_id, output=archive_path, quiet=False)
        print("\nDownload finished!")
    except Exception as e:
        traceback.print_exc()
        print("Download failed.")
        return

    # Optional MD5 check
    if expected_md5 is not None:
        print("Verifying MD5 checksum...")
        md5_val = calculate_md5(archive_path)
        assert md5_val == expected_md5, f"MD5 mismatch: {md5_val} != {expected_md5}"
        print("MD5 checksum passed!")

    print("Extracting archive...")
    extract_target = os.path.join(output_folder, "sketches")

    try:
        with py7zr.SevenZipFile(archive_path, mode="r") as zf:
            zf.extractall(path=output_folder)
    except Exception:
        traceback.print_exc()
        print("Extraction failed.")
        return

    # Expect a folder named "sketches" after extraction.
    if not os.path.isdir(extract_target):
        raise FileNotFoundError(
            f'Expected directory "{extract_target}" not found after extraction.'
        )

    print("Extraction finished!")

    # Optionally remove invalid sketches listed in per-class invalid.txt files
    if prune_invalid:
        try:
            pruned_total, pruned_detail = prune_invalid_svgs(extract_target)
            print(f"Pruned {pruned_total} invalid SVGs as listed by invalid.txt files.")
        except Exception:
            traceback.print_exc()
            print("Warning: failed to prune invalid SVGs; continuing.")

    if remove_sourcefile:
        try:
            os.remove(archive_path)
            print("Removed source archive.")
        except Exception as e:
            print(f"Warning: could not remove archive: {e}")


# Helper function to prune SVGs listed in invalid.txt per class
def prune_invalid_svgs(sketches_dir: str) -> tuple[int, dict]:
    """Delete SVGs listed in each class folder's ``invalid.txt``.

    Assumes every class folder contains ``invalid.txt`` with one basename per line
    (without extension), and corresponding files always exist as ``.svg``.
    """
    total_deleted = 0
    per_class_counts: dict[str, int] = {}

    if not os.path.isdir(sketches_dir):
        raise FileNotFoundError(f"Directory not found: {sketches_dir}")

    for class_name in sorted(os.listdir(sketches_dir)):
        class_dir = os.path.join(sketches_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        invalid_path = os.path.join(class_dir, "invalid.txt")
        deleted_here = 0

        try:
            with open(invalid_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            traceback.print_exc()
            continue

        for base in lines:
            # Skip headings or section markers
            if base.lower() in {
                "invalid/error",
                "unidentifiable/ambiguous",
                "context",
                "perspective",
            }:
                continue

            target = os.path.join(class_dir, f"{base}.svg")
            try:
                if os.path.exists(target):
                    os.remove(target)
                    deleted_here += 1
                    total_deleted += 1
            except Exception:
                traceback.print_exc()

        if deleted_here > 0:
            per_class_counts[class_name] = deleted_here

    return total_deleted, per_class_counts


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
            # Convert a line segment to a cubic by interpolation between start and end.
            # Do not assume a previous Move; add the start anchor if missing.
            if len(control_points_list) == 0:
                s = path_item.start
                control_points_list.append((s.real, s.imag))
            start, end = path_item.start, path_item.end
            start_x, start_y = start.real, start.imag
            end_x, end_y = end.real, end.imag

            control1_x = 2.0 / 3.0 * start_x + 1.0 / 3.0 * end_x
            control1_y = 2.0 / 3.0 * start_y + 1.0 / 3.0 * end_y
            control2_x = 1.0 / 3.0 * start_x + 2.0 / 3.0 * end_x
            control2_y = 1.0 / 3.0 * start_y + 2.0 / 3.0 * end_y

            control_points_list.append((control1_x, control1_y))
            control_points_list.append((control2_x, control2_y))
            control_points_list.append((end_x, end_y))

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


class Sketchy:
    """The Sketchy Database (https://sketchy.eye.gatech.edu/), SVG subset.

    This loader expects the archive ``sketches-06-04.7z`` to extract into a
    ``sketches/`` directory whose immediate subfolders are category names,
    each containing multiple ``.svg`` files. Any ``checked.txt`` or
    ``invalid.txt`` files present in category folders are ignored.

    After extraction, SVGs listed in per-class ``invalid.txt`` files are
    automatically deleted (this behavior can be disabled via
    ``prune_invalid=False`` in ``download_sketchy_database``).

    The data is represented using cubic Bézier curves consistent with SketchKit.

    Example:
        >>> data = SketchyDatabase()
        >>> len(data)
        20000
        >>> s = data[0]
        >>> isinstance(s, Sketch)
        True

    Args:
        data_base: Relative base under $HOME where the dataset will be stored.
            If the expected directory does not exist, it will be downloaded
            and extracted automatically.
        gdrive_url: Direct Google Drive download URL for the archive.
        archive_name: Local filename for the 7z archive.
        md5: Optional MD5 checksum string to verify archive integrity.
        remove_sourcefile: If True, delete the archive after extraction.
    """

    def __init__(
        self,
        data_base: str = ".cache/sketchkit/datasets",
        file_id: str = "1Qr8HhjRuGqgDONHigGszyHG_awCstivo",
        archive_name: str = "sketches-06-04.7z",
        md5: str | None = None,
        remove_sourcefile: bool = True,
        prune: bool = True,
    ):
        data_root = os.path.join(os.environ.get("HOME"), data_base)
        os.makedirs(data_root, exist_ok=True)

        # Download & extract if needed.
        if not os.path.isdir(os.path.join(data_root, "sketches")):
            download_sketchy_database(
                output_folder=data_root,
                remove_sourcefile=remove_sourcefile,
                expected_md5=md5,
                gdrive_id=file_id,
                filename=archive_name,
                prune_invalid=prune,
            )

        # Build index of classes and svg files.
        self.svg_base = os.path.join(data_root, "sketches")
        if not os.path.isdir(self.svg_base):
            raise FileNotFoundError(f'Expected directory "{self.svg_base}" not found.')

        class_names = [
            d
            for d in os.listdir(self.svg_base)
            if os.path.isdir(os.path.join(self.svg_base, d))
        ]
        class_names.sort()
        self.class_names = class_names

        sketch_names = []
        for class_name in self.class_names:
            class_base = os.path.join(self.svg_base, class_name)
            files = os.listdir(class_base)
            # Ignore marker TXT files, keep only .svg
            svg_files = sorted([f for f in files if f.lower().endswith(".svg")])
            sketch_names += [class_name + "/" + f for f in svg_files]

        if len(sketch_names) == 0:
            raise RuntimeError(
                f"No SVG files found under '{self.svg_base}'. "
                "Please ensure the correct archive is downloaded."
            )

        self.sketch_names = sketch_names

    def __len__(self) -> int:
        return len(self.sketch_names)

    def __getitem__(self, idx: int) -> Sketch:
        """Load a single sketch as a `Sketch` instance.

        Args:
            idx: Index into the dataset.

        Returns:
            A `Sketch` object composed of cubic Bézier `Path`/`Curve`s.
        """
        sketch_name = self.sketch_names[idx]  # e.g., "cat/12345.svg"
        class_name, file_name = sketch_name.split("/")
        svg_path = os.path.join(self.svg_base, class_name, file_name)

        # Parse original SVG into SketchVG-compatible parts
        image_sizes, path_list, total_segment_num = parse_svg(svg_path)
        # image_sizes: (width, height)
        # path_list: list of (N_curves, 4, 2)

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
