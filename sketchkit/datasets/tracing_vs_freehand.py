import os
import shutil
import wget
import zipfile
import xml.etree.ElementTree as ET
from svg.path import parse_path, path
import numpy as np

from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point

def download_and_extract_data(output_folder: str):
    """Downloads and extracts the Tracing-vs-Freehand dataset.

    Downloads a zip file from a predefined URL, extracts its contents to a
    structured directory, and cleans up the downloaded zip file.

    Args:
        output_folder: The base directory where datasets are stored. The
            data will be extracted into a 'tracing_vs_freehand' subdirectory
            within this folder.
    """
    url = "https://cislab.hkust-gz.edu.cn/projects/tracing-vs-freehand/data/sketch.zip"
    zip_path = os.path.join(output_folder, "sketch.zip")
    final_data_path = os.path.join(output_folder, "tracing_vs_freehand")

    os.makedirs(output_folder, exist_ok=True)
    print("Downloading the Tracing-vs-Freehand dataset...")
    try:
        wget.download(url, out=zip_path)
        print("\nDownload finished!")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return

    print("Extracting files...")
    try:
        temp_extract_path = os.path.join(output_folder, "temp_extract")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_path)

        # The zip archive contains a root 'data' folder; move it to the final path.
        extracted_root = os.path.join(temp_extract_path, "data")
        shutil.move(extracted_root, final_data_path)
        print("Extraction complete!")
    except Exception as e:
        print(f"Extraction failed: {e}")
    finally:
        # Clean up the downloaded zip and temporary directories
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(temp_extract_path):
            shutil.rmtree(temp_extract_path)

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


class TracingVsFreehand:
    """Loads the Tracing-vs-Freehand dataset from SVG files.

    This class handles the automatic download, extraction, and parsing of the
    Tracing-vs-Freehand dataset. It indexes all SVG sketches and provides an
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
        self.data_root = os.path.join(data_base, "tracing_vs_freehand")
        if not os.path.isdir(self.data_root):
            print(f"Dataset not found at '{self.data_root}'.")
            download_and_extract_data(data_base)

        self.svg_base = os.path.join(self.data_root, "svg")
        if not os.path.isdir(self.svg_base):
            raise FileNotFoundError(
                f"SVG directory not found at '{self.svg_base}'. "
                "Please check the download or extraction process."
            )

        self.sketch_paths = []
        sketch_types = ["drawings", "drawings_registered", "tracings"]
        for sketch_type in sketch_types:
            type_dir = os.path.join(self.svg_base, sketch_type)
            if not os.path.isdir(type_dir):
                print(f"Warning: Directory for type '{sketch_type}' not found, skipping. Path: {type_dir}")
                continue
            for filename in sorted(os.listdir(type_dir)):
                if filename.endswith(".svg"):
                    self.sketch_paths.append(os.path.join(type_dir, filename))

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
                strokes.append(Stroke(p_start, p_end, p_ctrl1, p_ctrl2))
            paths.append(Path(strokes=strokes))

        sketch = Sketch(paths=paths)
        return sketch
