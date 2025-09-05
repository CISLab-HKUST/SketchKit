import os
import shutil
import wget
import zipfile
import xml.etree.ElementTree as ET
from svg.path import parse_path, path
import numpy as np
import traceback
import gdown
from tqdm import tqdm
import pandas as pd
from pathlib import Path as PathLibPath

from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import (
    download_with_tqdm,
    CISLAB_CDN,
    extract_files,
    file_md5,
)

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

        view_box = root.get("viewBox")
        view_x, view_y, view_width, view_height = view_box.split(",")
        assert int(view_x) == 0 and int(view_y) == 0
        sketch_width = int(view_width) - int(view_x)
        sketch_height = int(view_height) - int(view_y)
        
        path_list = []
        for elem in root.findall('svg:path', namespaces):
            path_d = elem.attrib.get('d', '')
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
                    p_end   = control_points[i * 3 + 3]
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

    
    def __init__(
        self,
        root: str | PathLibPath | None = None,
        load_all: bool = False,
        cislab_source: bool = False,
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
        - sub_id: Identifier within the category and split
        """
        if not os.path.exists(os.path.join(self.root, ".metadata.parquet")):
            svg_base = os.path.join(self.root, "opensketch_svgs")
            # all_categories = os.listdir(svg_base)
            all_categories = ['svg']
            self._all_categories = sorted(all_categories)

            items_metadata = pd.DataFrame(columns=["category", "split", "id", "sub_id"])
            cnt = 0
            for category in tqdm(self._all_categories, desc="Caching metadata"):
                class_base = os.path.join(self.root, "opensketch_svgs")
                file_names = os.listdir(class_base)
                file_ids = [fid[:-4] for fid in file_names]
                file_ids.sort()

                split = "None"

                new_rows = [
                    {
                        "category": category,
                        "split": split,
                        "id": cnt + id,
                        "sub_id": file_id,
                    }
                    for id, file_id in enumerate(file_ids)
                ]
                cnt += len(file_ids)
                items_metadata = pd.concat(
                    [items_metadata, pd.DataFrame(new_rows)], ignore_index=True
                )
            self.items_metadata = items_metadata
            self.items_metadata.to_parquet(
                os.path.join(self.root, ".metadata.parquet"), compression="zstd"
            )
        else:
            items_metadata = pd.read_parquet(
                os.path.join(self.root, ".metadata.parquet")
            )
            if not set(items_metadata.columns.tolist()) == set(self.items_metadata):
                os.remove(os.path.join(self.root, ".metadata.parquet"))
                self._load_items_metadata()
            self.items_metadata = items_metadata
            self._all_categories = sorted(
                self.items_metadata["category"].unique().tolist()
            )
        self.raw_data = [None] * len(self.items_metadata)  # type: list[list | None]

    def _load_all(self):
        """Load all sketch data into memory if load_all is enabled.

        Concatenates all sketch data from all categories and splits into a single
        numpy array for faster access. Only loads if self.load_all is True.
        """
        cnt = 0
        for category in tqdm(self._all_categories, desc="Loading all data to memory"):
            class_base = os.path.join(self.root, "opensketch_svgs")
            file_names = os.listdir(class_base)
            file_ids = [fid[:-4] for fid in file_names]
            file_ids.sort()

            for file_id in file_ids:
                svg_path = os.path.join(class_base, str(file_id) + ".svg")
                _, path_list = parse_svg(svg_path)
                # path_list: list of (N_curves, 4, 2)
                self.raw_data[cnt] = path_list
                cnt += 1

    # def __init__(self, data_base: str = os.path.join(os.path.expanduser("~"), ".cache/sketchkit/datasets")):
    #     """Initializes the dataset loader.

    #     Checks if the dataset exists locally. If not, it triggers the download
    #     and extraction process. It then scans the SVG directories to populate a
    #     list of all available sketch file paths.

    #     Args:
    #         data_base: The base directory for storing all datasets.
    #             Defaults to '~/.cache/sketchkit/datasets'.
    #     """
    #     self.data_root = os.path.join(data_base, "opensketch")
    #     if not os.path.isdir(self.data_root):
    #         os.makedirs(self.data_root, exist_ok=True)
    #         print(f"Dataset not found at '{self.data_root}'.")
    #         download_and_extract_data(self.data_root)

    #     self.svg_base = os.path.join(self.data_root, "opensketch_svgs")
    #     if not os.path.isdir(self.svg_base):
    #         raise FileNotFoundError(
    #             f"SVG directory not found at '{self.svg_base}'. "
    #             "Please check the download or extraction process."
    #         )

    #     self.sketch_paths = []
    #     for filename in sorted(os.listdir(self.svg_base)):
    #         if filename.endswith(".svg"):
    #             self.sketch_paths.append(os.path.join(self.svg_base, filename))

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
            svg_path = os.path.join(class_base, str(item_metadata["sub_id"]) + ".svg")
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
    dataset = OpenSketch()

    # # Load all data to memory at once
    # dataset = OpenSketch(load_all=True)
    
    # Download from CISLAB CDN
    dataset = OpenSketch(cislab_source=True)

    # Search items using metadata
    objects = dataset.items_metadata[
        (dataset.items_metadata["category"] == "svg")
        & (dataset.items_metadata["split"] == "None")
    ]

    # Load sketches
    sketches = [dataset[row.id] for _, row in objects[:5].iterrows()]

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer

    renderer = CairoRenderer(800, (1, 1, 1))
    raster_image = renderer.render(sketches[1])
    outpath = "opensketch-new.png"
    if outpath is not None:
        raster_image_png = Image.fromarray(raster_image, "RGB")
        raster_image_png.save(outpath, "PNG")
