import os
import shutil
import re
import math

import xml.etree.ElementTree as ET

from svg.path import parse_path, path

from tqdm import tqdm
import pandas as pd


from sketchkit.core.sketch import Sketch, Path, Curve, Point
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import (
    download_with_wget,
    extract_files,
    file_md5,
)


invalid_svg_shapes = ["rect", "circle", "ellipse", "line", "polyline", "polygon"]


def parse_single_path(path_str):
    try:
        ps = parse_path(path_str)
    except Exception as e:
        print(e)
        return None

    control_points_list = []
    for item_i, path_item in enumerate(ps):
        path_type = type(path_item)

        if path_type == path.Move:
            assert item_i == 0
            start = path_item.start
            start_x, start_y = start.real, start.imag
            control_points_list.append((start_x, start_y))
        elif path_type == path.CubicBezier:
            start, control1, control2, end = (
                path_item.start,
                path_item.control1,
                path_item.control2,
                path_item.end,
            )
            start_x, start_y = start.real, start.imag
            control1_x, control1_y = control1.real, control1.imag
            control2_x, control2_y = control2.real, control2.imag
            end_x, end_y = end.real, end.imag
            control_points_list.append((control1_x, control1_y))
            control_points_list.append((control2_x, control2_y))
            control_points_list.append((end_x, end_y))
        elif path_type == path.Arc:
            raise Exception("Arc is not supported")
        elif path_type == path.Line:
            assert len(control_points_list) == 1
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


# def parse_transform(svg_transform):
#     # Regular expression to find translate and scale commands
#     pattern = r"(\w+)\s*\(([^)]+)\)"
#
#     # Find all matches
#     matches = re.findall(pattern, svg_transform)
#
#     # Parse the matches into a dictionary
#     transformations = []
#     for command, values in matches:
#         # Split the values by comma and convert to float
#         values = tuple(map(float, values.split(",")))
#         transformations.append([command, values])
#         assert command in ["translate", "scale"], command
#
#     return transformations

class TransformMatrix:
    def __init__(self):
        # Initialize as the identity matrix
        self.a, self.b, self.c, self.d, self.e, self.f = 1, 0, 0, 1, 0, 0

    def multiply(self, a, b, c, d, e, f):
        new_a = self.a * a + self.c * b
        new_b = self.b * a + self.d * b
        new_c = self.a * c + self.c * d
        new_d = self.b * c + self.d * d
        new_e = self.a * e + self.c * f + self.e
        new_f = self.b * e + self.d * f + self.f
        self.a, self.b, self.c, self.d, self.e, self.f = new_a, new_b, new_c, new_d, new_e, new_f

    def translate(self, tx, ty):
        self.multiply(1, 0, 0, 1, tx, ty)

    def scale(self, sx, sy=None):
        if sy is None:
            sy = sx
        self.multiply(sx, 0, 0, sy, 0, 0)

    def rotate(self, angle, cx=None, cy=None):
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        if cx is not None and cy is not None:
            self.translate(cx, cy)
            self.multiply(cos_a, sin_a, -sin_a, cos_a, 0, 0)
            self.translate(-cx, -cy)
        else:
            self.multiply(cos_a, sin_a, -sin_a, cos_a, 0, 0)

    def skew_x(self, angle):
        rad = math.radians(angle)
        tan_a = math.tan(rad)
        self.multiply(1, 0, tan_a, 1, 0, 0)

    def skew_y(self, angle):
        rad = math.radians(angle)
        tan_a = math.tan(rad)
        self.multiply(1, tan_a, 0, 1, 0, 0)

    def matrix(self, a, b, c, d, e, f):
        self.multiply(a, b, c, d, e, f)

    def apply_to_point(self, x, y):
        new_x = self.a * x + self.c * y + self.e
        new_y = self.b * x + self.d * y + self.f
        return new_x, new_y


def parse_transform(transform_str):
    matrix = TransformMatrix()

    transforms = re.findall(r'(\w+)\(([^)]+)\)', transform_str)

    for transform_type, params_str in transforms:
        params = [float(p) for p in re.split(r'[,\s]+', params_str) if p]

        if transform_type == 'translate':
            if len(params) == 1:
                matrix.translate(params[0], 0)
            else:
                matrix.translate(params[0], params[1])

        elif transform_type == 'scale':
            if len(params) == 1:
                matrix.scale(params[0])
            else:
                matrix.scale(params[0], params[1])

        elif transform_type == 'rotate':
            if len(params) == 1:
                matrix.rotate(params[0])
            else:
                matrix.rotate(params[0], params[1], params[2])

        elif transform_type == 'skewX':
            matrix.skew_x(params[0])

        elif transform_type == 'skewY':
            matrix.skew_y(params[0])

        elif transform_type == 'matrix':
            matrix.matrix(*params)

    return matrix


def apply_transform_to_path(path_d, matrix):
    path_pattern = re.compile(r'([a-zA-Z])([^a-zA-Z]*)')
    commands = path_pattern.findall(path_d)

    new_d = []
    current_point = (0, 0)

    for cmd, params_str in commands:
        params = [float(p) for p in re.split(r'[,\s]+', params_str) if p]

        if cmd in 'MmLl':
            i = 0
            while i < len(params):
                x, y = params[i], params[i + 1]
                if cmd.islower():
                    x += current_point[0]
                    y += current_point[1]
                new_x, new_y = matrix.apply_to_point(x, y)
                new_d.append(f'{cmd}{new_x:.6f},{new_y:.6f}')
                current_point = (x, y)
                i += 2

        elif cmd in 'CcSsQqTt':
            if cmd in 'CcSs':
                param_count = 6
            else:
                param_count = 4

            i = 0
            while i < len(params):
                segment_params = params[i:i + param_count]
                transformed_params = []

                for j in range(0, len(segment_params), 2):
                    x, y = segment_params[j], segment_params[j + 1]
                    if cmd.islower():
                        x += current_point[0]
                        y += current_point[1]
                    new_x, new_y = matrix.apply_to_point(x, y)
                    transformed_params.extend([new_x, new_y])

                    if j == len(segment_params) - 2:
                        current_point = (x, y)

                params_str = ','.join([f'{p:.6f}' for p in transformed_params])
                new_d.append(f'{cmd}{params_str}')
                i += param_count

        elif cmd in 'Aa':
            i = 0
            while i < len(params):
                rx, ry, x_axis_rotation, large_arc_flag, sweep_flag, x, y = params[i:i + 7]

                if cmd.islower():
                    x += current_point[0]
                    y += current_point[1]

                new_x, new_y = matrix.apply_to_point(x, y)

                new_d.append(
                    f'{cmd}{rx:.6f},{ry:.6f},{x_axis_rotation:.6f},{large_arc_flag},{sweep_flag},{new_x:.6f},{new_y:.6f}')
                current_point = (x, y)
                i += 7

        elif cmd in 'HhVv':
            is_horizontal = cmd in 'Hh'

            for p in params:
                if is_horizontal:
                    x = p
                    y = current_point[1]
                    if cmd.islower():
                        x += current_point[0]
                else:
                    x = current_point[0]
                    y = p
                    if cmd.islower():
                        y += current_point[1]

                new_x, new_y = matrix.apply_to_point(x, y)
                new_d.append(f'{cmd}{new_x:.6f}' if is_horizontal else f'{cmd}{new_y:.6f}')
                current_point = (x, y)

        elif cmd == 'Z' or cmd == 'z':
            new_d.append(cmd)

    return ' '.join(new_d)


def parse_svg(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()

    view_box = root.get("viewBox")
    view_x, view_y, view_width, view_height = view_box.split(" ")
    assert int(view_x) == 0 and int(view_y) == 0
    sketch_width = int(view_width) - int(view_x)
    sketch_height = int(view_height) - int(view_y)

    path_list = []  # list of (N_curves, 4, 2)
    matrix = None

    for elem in root.iter():
        try:
            _, tag_suffix = elem.tag.split("}")
        except ValueError:
            continue

        assert tag_suffix not in invalid_svg_shapes

        if tag_suffix == "g":
            if "transform" not in elem.attrib.keys():
                continue

            transform_str = elem.attrib['transform']
            matrix = parse_transform(transform_str)

        elif tag_suffix == "path":
            assert "transform" not in elem.attrib.keys()

            path_d = elem.attrib["d"]
            if matrix is not None:
                path_d = apply_transform_to_path(path_d, matrix)

            control_points_list = parse_single_path(path_d)  # (N, 2)
            if control_points_list is not None:
                segment_num = (len(control_points_list) - 1) // 3
                curve_list = []
                for i in range(segment_num):
                    curve = control_points_list[i * 3 : i * 3 + 4]  # (4, 2)
                    curve_list.append(curve)
                path_list.append(curve_list)

    assert len(path_list) > 0
    return (sketch_width, sketch_height), path_list


class TUBerlin(SketchDataset):
    """The TU-Berlin dataset contains vector sketches represented with cubic Bézier curves across 250 categories.
    Each category contains 800 sketches.

    Attributes:
        md5_sum (str): MD5 checksum for dataset integrity verification.

    References:
        - Original dataset: http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/
    """

    # MD5 Sum of the whole dir, you set a list for zip files
    md5_sum = "aa7d8ae9c8bf5f5cb6d28cee9741737c"

    # Metadata columns for each items
    metadata = ["id", "sub_id", "category", "split"]

    # def __init__(
    #     self,
    #     root: str | PathLibPath | None = None,
    #     load_all: bool = False,
    # ):
    #     """Initialize the TU-Berlin dataset.

    #     Args:
    #         root (str | PathLibPath | None): Root directory to store/load the dataset.
    #             If None, uses the default dataset directory.
    #         load_all (bool): If True, loads all data into memory for faster access.
    #             If False, loads data on-demand. Defaults to False.
    #             If False, downloads from original Google storage. Defaults to False.
    #     """

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
        """Download the TU-Berlin dataset.

        Downloads the archive from the original host. Creates the root directory
        if it doesn't exist.

        Raises:
            Exception: If download fails for any file.
        """

        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

        svg_zip_path = os.path.join(self.root, "sketches_svg.zip")
        try:
            url = "http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_svg.zip"
            download_with_wget(url, file_path=svg_zip_path)
        except Exception as e:
            raise e

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
            svg_base = os.path.join(self.root, "svg")
            all_categories = os.listdir(svg_base)
            all_categories = [
                item
                for item in all_categories
                if os.path.isdir(os.path.join(svg_base, item))
            ]
            self.all_categories = sorted(all_categories)

            items_metadata = pd.DataFrame(columns=["category", "split", "id", "sub_id"])
            cnt = 0
            for category in tqdm(self.all_categories, desc="Caching metadata"):
                class_base = os.path.join(self.root, "svg", category)
                file_names = os.listdir(class_base)
                file_ids = [int(item[:-4]) for item in file_names]
                file_ids.sort()

                valid_test_num = len(file_ids) // (10 + 1 + 1)
                train_num = len(file_ids) - valid_test_num - valid_test_num
                split_map = ["train"] * train_num + ["valid"] * valid_test_num + ["test"] * valid_test_num

                new_rows = [
                    {
                        "category": category,
                        "split": split_map[id],
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
            if not set(items_metadata.columns.tolist()) == set(self.metadata):
                os.remove(os.path.join(self.root, ".metadata.parquet"))
                return self._load_items_metadata()
            self.items_metadata = items_metadata
            self.all_categories = sorted(
                self.items_metadata["category"].unique().tolist()
            )
        self.raw_data = [None] * len(self.items_metadata)  # type: list[list | None]

    def _load_all(self):
        """Load all sketch data into memory if load_all is enabled.

        Concatenates all sketch data from all categories and splits into a single
        numpy array for faster access. Only loads if self.load_all is True.
        """
        cnt = 0
        for category in tqdm(self.all_categories, desc="Loading all data to memory"):
            class_base = os.path.join(self.root, "svg", category)
            file_names = os.listdir(class_base)
            file_ids = [int(item[:-4]) for item in file_names]
            file_ids.sort()

            for file_id in file_ids:
                svg_path = os.path.join(class_base, str(file_id) + ".svg")
                (sketch_width, sketch_height), path_list = parse_svg(svg_path)
                # path_list: list of (N_curves, 4, 2)
                self.raw_data[cnt] = [path_list, (sketch_width, sketch_height)]
                cnt += 1

    def _get_single(self, idx: int) -> Sketch:
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
            class_base = os.path.join(self.root, "svg", item_metadata["category"])
            svg_path = os.path.join(class_base, str(item_metadata["sub_id"]) + ".svg")
            (sketch_width, sketch_height), path_list = parse_svg(svg_path)
            self.raw_data[idx] = [path_list, (sketch_width, sketch_height)]

        path_list, (width, height) = self.raw_data[idx]
        # path_list: list of (N_curves, 4, 2)

        paths = []
        for path in path_list:
            curves = []
            for raw_curve in path:
                p_start = Point(raw_curve[0][0], raw_curve[0][1])
                p_end = Point(raw_curve[3][0], raw_curve[3][1])
                p_ctrl1 = Point(raw_curve[1][0], raw_curve[1][1])
                p_ctrl2 = Point(raw_curve[2][0], raw_curve[2][1])
                curves.append(Curve(p_start, p_end, p_ctrl1, p_ctrl2))
            paths.append(Path(curves=curves))
        sketch = Sketch(height=height, width=width, paths=paths)
        return sketch

    def extra_repr(self) -> str:
        num_train = len(self.items_metadata[self.items_metadata["split"] == "train"])
        num_test = len(self.items_metadata[self.items_metadata["split"] == "test"])
        num_val = len(self.items_metadata[self.items_metadata["split"] == "valid"])
        return f"Categories: {len(self.all_categories)}\nTrain: {num_train} Validation: {num_val} Test samples: {num_test} "


if __name__ == "__main__":
    # Default Load
    dataset = TUBerlin()

    # # Load all data to memory at once
    # dataset = TUBerlin(load_all=True)
    #
    # Search items using metadata
    objects = dataset.items_metadata[
        (dataset.items_metadata["category"] == "airplane")
        & (dataset.items_metadata["split"] == "None")
    ]

    # Load sketches
    sketches = [dataset[row.id] for _, row in objects[:5].iterrows()]

    from sketchkit.renderer.cairo_renderer import CairoRenderer

    renderer = CairoRenderer()
    raster_image = renderer.render(sketches[0])
    outpath = "test.png"
    if outpath is not None:
        raster_image.save(outpath, "PNG")
