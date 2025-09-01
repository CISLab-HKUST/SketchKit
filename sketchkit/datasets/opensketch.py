import os
import shutil
import wget
import zipfile
import xml.etree.ElementTree as ET
from svg.path import parse_path, path
import numpy as np
import traceback
import gdown

from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point

# https://drive.google.com/file/d/1wZf3lkqSsqYTIdGqT0wrdryCcfl4f731/view?usp=sharing
def download_and_extract_data_p(output_folder: str):
    """Downloads and extracts the OpenSketch dataset.

    Downloads a zip file from a predefined URL, extracts its contents to a
    structured directory, and cleans up the downloaded zip file.

    Args:
        output_folder: The base directory where datasets are stored. The
            data will be extracted into a 'opensketch' subdirectory
            within this folder.
    """
    url = "https://repo-sam.inria.fr/d3/OpenSketch/OpenSketch.zip"
    zip_path = os.path.join(output_folder, "OpenSketch.zip")

    print("Downloading the OpenSketch dataset...")
    try:
        wget.download(url, out=zip_path)
        print("\nDownload finished!")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return

    print("Extracting files...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)

        temp_extract_path = os.path.join(output_folder, "all_zip")
        svg_root = os.path.join(temp_extract_path, "sketches_svgs.zip")
        print("svg_root: ", svg_root)
        with zipfile.ZipFile(svg_root, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
        print("Extraction complete!")
    except Exception as e:
        print(f"Extraction failed: {e}")
    finally:
        # Clean up the downloaded zip and temporary directories
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(temp_extract_path):
            shutil.rmtree(temp_extract_path)

def download_and_extract_data(
    output_folder: str,
    remove_sourcefile: bool = True,
    expected_md5: str | None = None,
    gdrive_id: str = "1wZf3lkqSsqYTIdGqT0wrdryCcfl4f731",
    filename: str = "opensketch_svgs.zip",
    prune_invalid: bool = True,
):
    """Download and extract The OpenSketch Database (SVG version).

    Downloads the zip archive from Google Drive by file ID (preferred),
    verifies MD5 if provided, and extracts it into a ``sketches_svg/`` directory.

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
    print("Downloading OpenSketch Database...")

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
    extract_target = os.path.join(output_folder, "opensketch_svgs")

    try:
        # with py7zr.SevenZipFile(archive_path, mode="r") as zf:
        #     zf.extractall(path=output_folder)
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
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

    if remove_sourcefile:
        try:
            os.remove(archive_path)
            print("Removed source archive.")
        except Exception as e:
            print(f"Warning: could not remove archive: {e}")

def parse_single_path(path_str: str) -> list[tuple[float, float]] | None:
    """Parses an SVG path 'd' attribute string into Bézier control points.

    Converts various SVG path commands (Move, CubicBezier, Line) into a
    standardized list of control points for cubic Bézier curves.

    Args:
        path_str: The string from the 'd' attribute of an SVG <path> element.

    Returns:
        A list of (x, y) control points, or None if the path is invalid or empty.
    """
    ps = parse_path(path_str)
    control_points_list = []
    for item_i, path_item in enumerate(ps):
        path_type = type(path_item)
        if path_type == path.Move:
            assert item_i == 0, "Move command should only appear at the start"
            start = path_item.start
            control_points_list.append((start.real, start.imag))
        elif path_type == path.CubicBezier:
            c1, c2, end = path_item.control1, path_item.control2, path_item.end
            control_points_list.extend([(c1.real, c1.imag), (c2.real, c2.imag), (end.real, end.imag)])
        elif path_type == path.Line:
            start, end = path_item.start, path_item.end
            # Convert line to an equivalent cubic Bézier curve
            c1_x = (start.real * 2/3 + end.real * 1/3)
            c1_y = (start.imag * 2/3 + end.imag * 1/3)
            c2_x = (start.real * 1/3 + end.real * 2/3)
            c2_y = (start.imag * 1/3 + end.imag * 2/3)
            control_points_list.extend([(c1_x, c1_y), (c2_x, c2_y), (end.real, end.imag)])
        elif path_type == path.Arc:
            raise NotImplementedError("Arc path segments are not supported.")
        else:
            raise TypeError(f"Unknown path segment type: {path_type}")

    if not control_points_list or (len(control_points_list) - 1) % 3 != 0:
        return None
    return control_points_list


def parse_svg(svg_file: str) -> list[list[list[tuple[float, float]]]]:
    """Parses an entire SVG file to extract all path data.

    Reads an SVG file, finds all <path> elements, and uses parse_single_path
    to convert them into a list of strokes, where each stroke is represented
    by its four control points.

    Args:
        svg_file: The file path to the SVG file.

    Returns:
        A nested list representing the sketch. The structure is:
        list of paths -> list of strokes -> list of 4 control points.
        Returns an empty list if the file cannot be parsed.
    """
    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()
        namespaces = {'svg': 'http://www.w3.org/2000/svg'}
        path_list = []
        for elem in root.findall('svg:path', namespaces):
            path_d = elem.attrib.get('d', '')
            if not path_d:
                continue
            control_points = parse_single_path(path_d)
            if control_points:
                segment_num = (len(control_points) - 1) // 3
                stroke_list = []
                for i in range(segment_num):
                    p_start = control_points[i * 3]
                    p_ctrl1 = control_points[i * 3 + 1]
                    p_ctrl2 = control_points[i * 3 + 2]
                    p_end   = control_points[i * 3 + 3]
                    stroke_list.append([p_start, p_ctrl1, p_ctrl2, p_end])
                path_list.append(stroke_list)
        return path_list
    except ET.ParseError:
        print(f"Warning: Failed to parse SVG file: {svg_file}")
        return []


class OpenSketch:
    """Loads the OpenSketch dataset from SVG files.

    This class handles the automatic download, extraction, and parsing of the
    OpenSketch dataset. It indexes all SVG sketches and provides an
    interface to access them as `Sketch` objects. Accessing an item
    (e.g., `dataset[0]`) returns a single `Sketch` object.

    Attributes:
        data_root (str): The absolute path to the dataset's root directory.
        svg_base (str): The path to the subdirectory containing SVG files.
        sketch_paths (list[str]): A list of file paths for all indexed sketches.
    """
    def __init__(self, data_base: str = os.path.join(os.path.expanduser("~"), ".cache/sketchkit/datasets")):
        """Initializes the dataset loader.

        Checks if the dataset exists locally. If not, it triggers the download
        and extraction process. It then scans the SVG directories to populate a
        list of all available sketch file paths.

        Args:
            data_base: The base directory for storing all datasets.
                Defaults to '~/.cache/sketchkit/datasets'.
        """
        self.data_root = os.path.join(data_base, "opensketch")
        if not os.path.isdir(self.data_root):
            os.makedirs(self.data_root, exist_ok=True)
            print(f"Dataset not found at '{self.data_root}'.")
            download_and_extract_data(self.data_root)

        self.svg_base = os.path.join(self.data_root, "opensketch_svgs")
        if not os.path.isdir(self.svg_base):
            raise FileNotFoundError(
                f"SVG directory not found at '{self.svg_base}'. "
                "Please check the download or extraction process."
            )

        self.sketch_paths = []
        for filename in sorted(os.listdir(self.svg_base)):
            if filename.endswith(".svg"):
                self.sketch_paths.append(os.path.join(self.svg_base, filename))

    def __len__(self) -> int:
        """Returns the total number of sketches in the dataset."""
        return len(self.sketch_paths)

    def __getitem__(self, index: int) -> Sketch:
        """Retrieves a sketch by its index.

        Args:
            index: The index of the sketch to retrieve.

        Returns:
            The parsed sketch as a `Sketch` object.
        """
        svg_path = self.sketch_paths[index]
        raw_path_list = parse_svg(svg_path)

        paths = []
        for raw_path in raw_path_list:
            strokes = []
            for raw_stroke in raw_path:
                p_start = Vertex(raw_stroke[0][0], raw_stroke[0][1])
                p_end   = Vertex(raw_stroke[3][0], raw_stroke[3][1])
                p_ctrl1 = Point(raw_stroke[1][0], raw_stroke[1][1])
                p_ctrl2 = Point(raw_stroke[2][0], raw_stroke[2][1])
                strokes.append(Curve(p_start, p_end, p_ctrl1, p_ctrl2))
            paths.append(Path(curves=strokes))

        sketch = Sketch(paths=paths)
        return sketch
