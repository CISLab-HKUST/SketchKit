import os

import xml.etree.ElementTree as ET
from svg.path import parse_path, path
import traceback
import gdown
from tqdm import tqdm
import pandas as pd
from pathlib import Path as PathLibPath

from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import (
    download_with_wget,
    CISLAB_CDN,
    extract_files,
    file_md5,
)


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
            control_points_list.extend(
                [(c1.real, c1.imag), (c2.real, c2.imag), (end.real, end.imag)]
            )
        elif path_type == path.Line:
            start, end = path_item.start, path_item.end
            # Convert line to an equivalent cubic Bézier curve
            c1_x = start.real * 2 / 3 + end.real * 1 / 3
            c1_y = start.imag * 2 / 3 + end.imag * 1 / 3
            c2_x = start.real * 1 / 3 + end.real * 2 / 3
            c2_y = start.imag * 1 / 3 + end.imag * 2 / 3
            control_points_list.extend(
                [(c1_x, c1_y), (c2_x, c2_y), (end.real, end.imag)]
            )
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
    to convert them into a list of curves, where each curves is represented
    by its four control points.

    Args:
        svg_file: The file path to the SVG file.

    Returns:
        A nested list representing the sketch. The structure is:
        list of paths -> list of curves -> list of 4 control points.
        Returns an empty list if the file cannot be parsed.
    """
    try:
        tree = ET.parse(svg_file)
        root = tree.getroot()
        namespaces = {"svg": "http://www.w3.org/2000/svg"}

        view_box = root.get("viewBox")
        view_x, view_y, view_width, view_height = view_box.split(",")
        assert int(view_x) == 0 and int(view_y) == 0
        sketch_width = int(view_width) - int(view_x)
        sketch_height = int(view_height) - int(view_y)

        path_list = []
        for elem in root.findall("svg:path", namespaces):
            path_d = elem.attrib.get("d", "")
            if not path_d:
                continue
            control_points = parse_single_path(path_d)
            if control_points:
                segment_num = (len(control_points) - 1) // 3
                curve_list = []
                for i in range(segment_num):
                    p_start = control_points[i * 3]
                    p_ctrl1 = control_points[i * 3 + 1]
                    p_ctrl2 = control_points[i * 3 + 2]
                    p_end = control_points[i * 3 + 3]
                    curve_list.append([p_start, p_ctrl1, p_ctrl2, p_end])
                path_list.append(curve_list)
        assert len(path_list) > 0
        return (sketch_width, sketch_height), path_list
    except ET.ParseError:
        print(f"Warning: Failed to parse SVG file: {svg_file}")
        return []


class OpenSketch(SketchDataset):
    """The OpenSketch dataset contains vector sketches represented with polyline curves.

    Attributes:
        md5_sum (str): MD5 checksum for dataset integrity verification.

    References:
        - Original dataset: https://www-sop.inria.fr/reves/Basilic/2019/GSHPDB19/
    """

    # MD5 Sum of the whole dir, you set a list for zip files
    md5_sum = "e1e5ec89ac1345c87ae5d92ab90a0725"

    gdrive_id = "1wZf3lkqSsqYTIdGqT0wrdryCcfl4f731"

    metadata = ["id", "file_name"]

    def __init__(
        self,
        root: str | PathLibPath | None = None,
        load_all: bool = False,
        cislab_source: bool = True,
    ):
        """Initialize the OpenSketch dataset.

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
        zip_path = os.path.join(self.root, "sketches_svg.zip")
        if not os.path.exists(zip_path):
            return False
        current_md5 = file_md5(zip_path)
        return current_md5 == self.md5_sum

    def _download(self):
        """Download the OpenSketch dataset.

        Raises:
            Exception: If download fails for any file.
        """

        print("Downloading OpenSketch Database...")

        svg_zip_path = os.path.join(self.root, "sketches_svg.zip")

        if self.cislab_source:
            print("Using CISLAB CDN for download...")
            url = (
                f"{CISLAB_CDN}/datasets/{self.__class__.__name__}/" + "sketches_svg.zip"
            )
            download_with_wget(
                url,
                file_path=svg_zip_path,
                pg_bar=True,
            )
        else:
            try:
                print("Using gdown for Google Drive download...")
                # Prefer gdown when available, it handles confirm tokens and cookies internally.
                gdown.download(id=self.gdrive_id, output=svg_zip_path, quiet=False)
                print("\nDownload finished!")
            except Exception as e:
                traceback.print_exc()
                print("Download failed.")

        try:
            extract_files(
                file_path=svg_zip_path,
                output_dir=self.root,
                remove_sourcefile=False,
            )
        except Exception as e:
            raise e

    def _load_items_metadata(self):
        """Load and cache metadata for all items in the dataset.

        Creates a parquet file containing metadata for all sketches including
        category, split (train/valid/test), global ID, and sub ID within category.
        If metadata cache exists, loads from the cached file.

        The metadata DataFrame contains columns:
        - category: Drawing category name
        - split: Data split (train/valid/test)
        - id: Global unique identifier across all sketches
        """
        if not os.path.exists(os.path.join(self.root, ".metadata.parquet")):
            items_metadata = pd.DataFrame(columns=self.metadata)
            class_base = os.path.join(self.root, "opensketch_svgs")
            file_names = os.listdir(class_base)
            for id, file_name in tqdm(
                enumerate(sorted(file_names)),
                desc="Caching metadata",
                total=len(file_names),
            ):
                file_name = file_name[:-4]  # remove .svg
                new_row = [{"id": id, "file_name": file_name}]
                items_metadata = pd.concat(
                    [items_metadata, pd.DataFrame(new_row)], ignore_index=True
                )
            self.items_metadata = items_metadata
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
        self.raw_data = [None] * len(self.items_metadata)  # type: list[list | None]

    def _load_all(self):
        """Load all sketch data into memory if load_all is enabled.

        Concatenates all sketch data from all categories and splits into a single
        numpy array for faster access. Only loads if self.load_all is True.
        """
        class_base = os.path.join(self.root, "opensketch_svgs")
        for _, row in tqdm(
            self.items_metadata.iterrows(),
            desc="Loading all data to memory",
            total=len(self.items_metadata),
        ):
            svg_path = os.path.join(class_base, str(row["file_name"]) + ".svg")
            _, path_list = parse_svg(svg_path)
            # path_list: list of (N_curves, 4, 2)
            self.raw_data[row["id"]] = path_list

    def __getitem__(self, idx: int) -> Sketch:
        """Get a sketch by index.

        If a sketch not in memory, load all sketches in the same category from disk.

        Args:
            idx (int): Index of the sketch to retrieve.

        Returns:
            Sketch: A Sketch object containing the drawing data as paths.

        Raises:
            IndexError: If index is out of range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        # check if in memory, if not load from disk, along with all items in the same category to avoid I/O cost
        if self.raw_data[idx] is None:
            item_metadata = self.items_metadata.iloc[idx]
            class_base = os.path.join(self.root, "opensketch_svgs")
            svg_path = os.path.join(
                class_base, str(item_metadata["file_name"]) + ".svg"
            )
            _, path_list = parse_svg(svg_path)
            self.raw_data[idx] = path_list

        path_list = self.raw_data[idx]
        # path_list: list of (N_curves, 4, 2)

        paths = []
        for path in path_list:
            curves = []
            for raw_curve in path:
                p_start = Vertex(raw_curve[0][0], raw_curve[0][1])
                p_end = Vertex(raw_curve[3][0], raw_curve[3][1])
                p_ctrl1 = Point(raw_curve[1][0], raw_curve[1][1])
                p_ctrl2 = Point(raw_curve[2][0], raw_curve[2][1])
                curves.append(Curve(p_start, p_end, p_ctrl1, p_ctrl2))
            paths.append(Path(curves=curves))
        sketch = Sketch(paths=paths)
        return sketch


if __name__ == "__main__":
    # Default Load
    # dataset = OpenSketch()

    # # Load all data to memory at once
    # dataset = OpenSketch(load_all=True)

    # Download from CISLAB CDN
    dataset = OpenSketch(cislab_source=True, load_all=True)

    # Load sketches

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer

    renderer = CairoRenderer(800, (1, 1, 1))
    raster_image = renderer.render(dataset[100])
    outpath = "opensketch-test.png"
    if outpath is not None:
        raster_image_png = Image.fromarray(raster_image, "RGB")
        raster_image_png.save(outpath, "PNG")
