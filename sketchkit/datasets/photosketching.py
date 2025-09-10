import os

import gdown
import xml.etree.ElementTree as ET
from svg.path import parse_path, path
from pathlib import Path as PathLibPath
import pandas as pd
from tqdm import tqdm

from sketchkit.core import Sketch
from sketchkit.core.sketch import Path, Curve, Vertex, Point
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import download_with_wget, dir_md5, extract_files, CISLAB_CDN


# SVG parsing constants and functions
INVALID_SVG_SHAPES = {"rect", "circle", "ellipse", "line", "polyline", "polygon"}


def parse_photosketching_svg(svg_file):
    """Parse an SVG file into width, height, path list, total segment number, and path attributes.

    Args:
        svg_file: The path to the SVG file.

    Returns:
        A tuple containing:
        - (width, height): The dimensions of the SVG canvas.
        - path_list: A list of paths, where each path contains strokes.
        - total_segment_num: The total number of segments in the SVG file [Segment here is curve in sketchkit].
        - path_attributes: A list of dictionaries containing path attributes.

    Raises:
        Exception: If unsupported elements like transforms or arcs are encountered.
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # Parse dimensions
    view_box = root.get("viewBox")
    if view_box:
        view_x, view_y, view_w, view_h = [float(x) for x in view_box.split()]
        width = int(view_w - view_x)
        height = int(view_h - view_y)
    else:
        width = int(float(root.get("width", "0").replace("px", "")) or 0)
        height = int(float(root.get("height", "0").replace("px", "")) or 0)

    path_list = []
    path_attributes = []  # path in sketchkit has attributes like color and thickness
    total_segment_num = 0

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag in INVALID_SVG_SHAPES:
            continue
        if tag == "path":
            if "transform" in elem.attrib:
                raise Exception(f"Transform not supported in {svg_file}")

            # extract path attributes (only keep color and thickness)
            # convert color value to RGB array format
            stroke_color = elem.get("stroke", "#000000")
            if stroke_color == "#000" or stroke_color == "#000000":
                rgb_color = (0.0, 0.0, 0.0)  # black
            elif stroke_color == "#fff" or stroke_color == "#ffffff":
                rgb_color = (1.0, 1.0, 1.0)  # white
            elif stroke_color.startswith("#"):
                # handle other hexadecimal colors
                hex_color = stroke_color.lstrip("#")
                if len(hex_color) == 3:
                    hex_color = "".join([c + c for c in hex_color])
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                rgb_color = (r, g, b)
            else:
                rgb_color = (0.0, 0.0, 0.0)  # default black

            # original geometry parsing code
            d = elem.attrib.get("d")
            if not d:
                continue
            ps = parse_path(d)
            cps = []

            # extract attributes of current path (path in svg file)
            attrs = {
                "stroke": rgb_color,  # color of path in svg file
                "stroke-width": float(
                    elem.get("stroke-width", 1)
                ),  # thickness of <path> in svg file
                "t": elem.get(
                    "t", ""
                ),  # timestamp information of <path> in svg file, not used in sketchkit
                "id": elem.get("id", ""),  # id of <path> in svg file
            }

            for seg in ps:
                if isinstance(seg, path.Move):
                    if len(cps) > 1:
                        # start a new path
                        seg_num = (len(cps) - 1) // 3
                        strokes = [
                            cps[i * 3 : i * 3 + 4] for i in range(seg_num)
                        ]  # stroke here is curve in sketchkit, so stroke[s] here is path in sketchkit
                        path_list.append(strokes)
                        path_attributes.append(
                            attrs
                        )  # add corresponding attributes, the length of path_attributes is the same as path_list
                        total_segment_num += seg_num
                    cps = [(seg.start.real, seg.start.imag)]
                elif isinstance(seg, path.CubicBezier):
                    s, c1, c2, e = seg.start, seg.control1, seg.control2, seg.end
                    cps.append((c1.real, c1.imag))
                    cps.append((c2.real, c2.imag))
                    cps.append((e.real, e.imag))
                elif isinstance(seg, path.Line):
                    s, e = seg.start, seg.end
                    sx, sy, ex, ey = s.real, s.imag, e.real, e.imag
                    c1x = 2 / 3 * sx + 1 / 3 * ex
                    c1y = 2 / 3 * sy + 1 / 3 * ey
                    c2x = 1 / 3 * sx + 2 / 3 * ex
                    c2y = 1 / 3 * sy + 2 / 3 * ey
                    cps.append((c1x, c1y))
                    cps.append((c2x, c2y))
                    cps.append((ex, ey))
                elif isinstance(seg, path.Arc):
                    raise Exception("Arc not supported")
                else:
                    raise Exception("Unknown segment type", type(seg))

            # Handle the last path if there are remaining control points
            if len(cps) > 1:
                seg_num = (len(cps) - 1) // 3
                strokes = [cps[i * 3 : i * 3 + 4] for i in range(seg_num)]
                path_list.append(strokes)
                path_attributes.append(
                    attrs
                )  # add corresponding attributes, the length of path_attributes is the same as path_list
                total_segment_num += seg_num

    return (width, height), path_list, total_segment_num, path_attributes


class PhotoSketching(SketchDataset):
    """The PhotoSketching dataset loader.

    PhotoSketching is a dataset for photo-to-sketch generation, containing paired images
    and their corresponding sketch representations. The dataset includes:
    - 1,000 outdoor photos
    - 5,000 SVG format sketches (5 sketches per photo, with stroke timestamps)
    - 15,000 PNG format rendered sketches (each SVG sketch rendered with 3 different line widths)

    Attributes:
        md5_sum (str): MD5 checksum for dataset integrity verification.
        metadata (list): List of metadata column names for the dataset.
    """

    # MD5 of the dataset directory (excluding hidden files). Support multiple sources.
    md5_sum = "8f1b44299a3023276f169b658793a1ab"
    # Metadata columns for each item
    metadata = ["id", "svg_path", "png_path", "photo_path"]

    def __init__(
        self,
        root: str | PathLibPath | None = None,
        load_all: bool = False,
        cislab_source: bool = False,
    ):
        """Initialize the PhotoSketching dataset."""
        super().__init__(root, load_all=load_all, cislab_source=cislab_source)

    def _check_integrity(self) -> bool:
        """Check the integrity of the cached dataset using MD5 checksum."""

        print(f"Checking integrity of cached {self.__class__.__name__} dataset...")
        current_md5 = dir_md5(self.root)
        return current_md5 == self.md5_sum

    def _download(self):
        """Download the PhotoSketching dataset.

        Downloads the all-in-one.zip archive from either Google Drive or CISLAB CDN mirror.
        Extracts the archive to the root directory.
        """
        if os.path.exists(self.root):
            import shutil

            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

        archive_path = os.path.join(self.root, "all-in-one.zip")

        try:
            if self.cislab_source:
                # Use fixed mirror folder name "PhotoSketching"
                url = f"{CISLAB_CDN}/datasets/PhotoSketching/all-in-one.zip"
                download_with_wget(
                    url, archive_path, desc="Downloading PhotoSketching dataset"
                )
            else:
                # Use Google Drive download
                file_id = "1_AIxKnZXQms5Ezb-cEeVIDIoVG-eliHc"
                gdown.download(id=file_id, output=archive_path, quiet=False)

            print("Extracting archive...")
            extract_files(
                file_path=archive_path,
                output_dir=self.root,
                remove_sourcefile=True,
            )
            print("PhotoSketching dataset extracted successfully!")
            print("MD5 Checksum:", dir_md5(self.root))

        except Exception as e:
            print(f"Download failed: {e}")
            raise e

    def _load_items_metadata(self):
        """Load and cache metadata for all items in the dataset.

        Creates a pickle file containing metadata for all SVG sketches including
        file paths and unique IDs. If metadata cache exists, loads from the cached file.
        """
        metadata_file = os.path.join(self.root, ".metadata.pkl")

        if not os.path.exists(metadata_file):
            print("Building metadata index...")
            sketch_svg_dir = os.path.join(self.root, "sketch")

            # Find all SVG files
            svg_files = []
            for root_dir, dirs, files in os.walk(sketch_svg_dir):
                for file in files:
                    if file.lower().endswith(".svg"):
                        rel_path = os.path.relpath(
                            os.path.join(root_dir, file), sketch_svg_dir
                        )
                        svg_files.append(rel_path)

            svg_files.sort()

            # Create metadata DataFrame
            items_metadata = pd.DataFrame(
                {
                    "id": range(len(svg_files)),
                    "svg_path": [os.path.join("sketch", f) for f in svg_files],
                    "png_path": [
                        os.path.join("sketch-rendered", f.replace(".svg", ".png"))
                        for f in svg_files
                    ],
                }
            )
            self.items_metadata = items_metadata
            self.items_metadata.to_pickle(metadata_file)
            print(f"Indexed {len(svg_files)} SVG files")
        else:
            items_metadata = pd.read_pickle(metadata_file)
            if not set(items_metadata.columns.tolist()) == set(self.metadata):
                os.remove(metadata_file)
                return self._load_items_metadata()
            self.items_metadata = items_metadata

        self.raw_data = [None] * len(self.items_metadata)

    def _load_all(self):
        """Load all sketch data into memory if load_all is enabled."""
        print("Loading all sketches into memory...")
        for idx in tqdm(range(len(self.items_metadata)), desc="Loading sketches"):
            if self.raw_data[idx] is None:
                self.raw_data[idx] = self._load_single_sketch(idx)

    def _load_single_sketch(self, idx: int) -> Sketch:
        """Load a single sketch from disk."""
        item_metadata = self.items_metadata.iloc[idx]
        sketch_svg_path = os.path.join(self.root, item_metadata["svg_path"])

        if not os.path.exists(sketch_svg_path):
            raise FileNotFoundError(f"SVG file not found: {sketch_svg_path}")

        # Parse the SVG file and construct a Sketch object
        try:
            (w, h), path_list, _, path_attributes = parse_photosketching_svg(
                sketch_svg_path
            )

            # Validate that we have valid paths
            if not path_list:
                raise ValueError(f"No valid paths found in SVG file: {sketch_svg_path}")

            paths = []
            for i, p in enumerate(path_list):
                strokes = []
                for seg in p:
                    # Validate segment has exactly 4 control points
                    if len(seg) != 4:
                        raise ValueError(
                            f"Invalid segment in SVG: expected 4 points, got {len(seg)}"
                        )

                    p0 = Vertex(seg[0][0], seg[0][1])
                    p3 = Vertex(seg[3][0], seg[3][1])
                    c1 = Point(seg[1][0], seg[1][1])
                    c2 = Point(seg[2][0], seg[2][1])
                    strokes.append(Curve(p0, p3, c1, c2))

                # create Path
                path = Path(curves=strokes)

                # set Path level attributes (will be automatically passed to all Curves and Vertices)
                if i < len(path_attributes):
                    attrs = path_attributes[i]
                    path.color = attrs["stroke"]
                    path.thickness = attrs["stroke-width"]

                paths.append(path)

            # create Sketch
            sketch = Sketch(paths=paths)
            return sketch

        except Exception as e:
            raise Exception(f"Failed to parse SVG file {sketch_svg_path}: {str(e)}")

    def __getitem__(self, idx: int) -> Sketch:
        """Get a sketch by index."""
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # Check if in memory, if not load from disk
        if self.raw_data[idx] is None:
            self.raw_data[idx] = self._load_single_sketch(idx)

        return self.raw_data[idx]

    def extra_repr(self) -> str:
        """Return extra information for the string representation."""
        return f"SVG sketches: {len(self.items_metadata)}"


if __name__ == "__main__":
    import psutil
    from rich.console import Console
    import time

    console = Console()

    console.print("PhotoSketching Dataset Test", style="bold blue")
    console.print()

    console.print("1. Default Load", style="green")
    dataset = PhotoSketching()
    # show brief information of the dataset
    console.print(dataset)
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    console.print(f"Current memory usage: {memory_mb:.2f} MB")

    # Test loading a few sketches
    start_time = time.time()
    sketches = [dataset[i] for i in range(min(5, len(dataset)))]
    console.print(
        f"Loading {len(sketches)} sketches in {time.time() - start_time:.2f}s"
    )

    # Show sketch information
    for i, sketch in enumerate(sketches):
        console.print(f"Sketch {i}: {sketch.path_num} paths, {sketch.curve_num} curves")

    del dataset, sketches

    console.print()

    console.print("2. Load All", style="green")
    dataset = PhotoSketching(load_all=True)
    # show brief information of the dataset
    console.print(dataset)
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    console.print(f"Current memory usage: {memory_mb:.2f} MB")

    # Test loading sketches from memory
    start_time = time.time()
    sketches = [dataset[i] for i in range(min(10, len(dataset)))]
    console.print(
        f"Loading {len(sketches)} sketches from memory in {time.time() - start_time:.2f}s"
    )

    del dataset, sketches

    console.print()

    console.print("3. Load from CISLAB CDN", style="green")

    # tempfile can create a temporary directory
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = PhotoSketching(root=tmpdir, cislab_source=True)
        # dataset = PhotoSketching(cislab_source=True)
        # show brief information of the dataset
        console.print(dataset)
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        console.print(f"Current memory usage: {memory_mb:.2f} MB")

        start_time = time.time()
        sketches = [dataset[i] for i in range(min(5, len(dataset)))]
        console.print(
            f"Loading {len(sketches)} sketches in {time.time() - start_time:.2f}s"
        )

        del dataset, sketches

    console.print()
