import os
import wget

import xml.etree.ElementTree as ET
from svg.path import parse_path, path

from sketchkit.utils.general_utils.md5sum import *
from sketchkit.utils.general_utils.md5sum_table import *
from sketchkit.utils.general_utils.extract_files import *
from sketchkit.core.sketch import SketchVG


def download_tu_berlin(output_folder, remove_sourcefile=True):
    print("Downloading started......")

    os.makedirs(output_folder, exist_ok=True)
    try:
        wget.download("http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_svg.zip",
                      out=os.path.join(output_folder, "sketches_svg.zip"))
        # wget.download("http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip",
        #               out=os.path.join(output_folder, "sketches_png.zip"))
    except Exception as e:
        print(e)
        return

    print("TU-Berlin is downloaded!")

    assert calculate_md5(os.path.join(output_folder, "sketches_svg.zip")) == MD5SUM_TABLE[
        "sketches_svg.zip"], 'sketches_svg.zip md5 checksum error'
    # assert calculate_md5(os.path.join(output_folder, "sketches_png.zip")) == MD5SUM_TABLE[
    #     "sketches_png.zip"], 'sketches_png.zip md5 checksum error'

    print("Md5 checksum passed!")
    print("Extracting started......")

    try:
        extract_files(file_name=os.path.join(output_folder, "sketches_svg.zip"),
                      output_folder=os.path.join(output_folder, "tu_berlin"),
                      remove_sourcefile=remove_sourcefile)
        # extract_files(file_name=os.path.join(output_folder, "sketches_png.zip"),
        #               output_folder=os.path.join(output_folder, "tu_berlin"),
        #               remove_sourcefile=remove_sourcefile)

    except Exception as e:
        print(e)
        return

    print("TU-Berlin is extracted!")


invalid_svg_shapes = ['rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon']


def parse_single_path(path_str):
    ps = parse_path(path_str)

    control_points_list = []
    for item_i, path_item in enumerate(ps):
        path_type = type(path_item)

        if path_type == path.Move:
            assert item_i == 0
            start = path_item.start
            start_x, start_y = start.real, start.imag
            control_points_list.append((start_x, start_y))
        elif path_type == path.CubicBezier:
            start, control1, control2, end = path_item.start, path_item.control1, path_item.control2, path_item.end
            start_x, start_y = start.real, start.imag
            control1_x, control1_y = control1.real, control1.imag
            control2_x, control2_y = control2.real, control2.imag
            end_x, end_y = end.real, end.imag
            control_points_list.append((control1_x, control1_y))
            control_points_list.append((control2_x, control2_y))
            control_points_list.append((end_x, end_y))
        elif path_type == path.Arc:
            raise Exception('Arc is not supported')
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
            raise Exception('Unknown path_type', path_type)

    assert (len(control_points_list) - 1) % 3 == 0

    if len(control_points_list) > 1:
        return control_points_list
    else:
        return None


def parse_svg(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()

    view_box = root.get('viewBox')
    view_x, view_y, view_width, view_height = view_box.split(' ')
    assert int(view_x) == 0 and int(view_y) == 0
    sketch_width = int(view_width) - int(view_x)
    sketch_height = int(view_height) - int(view_y)

    path_list = []  # list of (N_strokes, 4, 2)
    total_segment_num = 0

    for elem in root.iter():
        try:
            _, tag_suffix = elem.tag.split('}')
        except ValueError:
            continue

        assert tag_suffix not in invalid_svg_shapes

        if tag_suffix == 'path':
            assert 'transform' not in elem.attrib.keys()

            path_d = elem.attrib['d']
            control_points_list = parse_single_path(path_d)  # (N, 2)
            if control_points_list is not None:
                segment_num = (len(control_points_list) - 1) // 3
                stroke_list = []
                for i in range(segment_num):
                    stroke = control_points_list[i * 3: i * 3 + 4]  # (4, 2)
                    stroke_list.append(stroke)
                path_list.append(stroke_list)
                total_segment_num += segment_num

    assert len(path_list) > 0
    return (sketch_width, sketch_height), path_list, total_segment_num


class TUBerlin:
    """ The TU-Berlin dataset (http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/),
    which contains vector sketches represented with cubic Bézier curves.

    Args:
        data_base: The directory of storing the dataset, which is automatically downloaded if not existed.

    Example:
        >>> my_data = TUBerlin()
        >>> print(len(my_data))
        20000
        >>> print(my_data[0])
        (<sketchkit.core.sketch.SketchVG object at 0x72a9fd79ae00>, 126)
    """

    def __init__(self, data_base=".cache/sketchkit/datasets"):
        data_root = os.path.join(os.environ.get('HOME'), data_base)
        os.makedirs(data_root, exist_ok=True)

        # First, download the dataset if not existed.
        if not os.path.isdir(os.path.join(data_root, "tu_berlin")):
            download_tu_berlin(data_root)

        # Then, load the dataset to obtain data like class names and all the file names.
        self.svg_base = os.path.join(data_root, "tu_berlin", "svg")
        class_names = os.listdir(self.svg_base)
        class_names = [item for item in class_names if os.path.isdir(os.path.join(self.svg_base, item))]
        class_names.sort()
        self.class_names = class_names

        sketch_names = []
        for class_name in self.class_names:
            class_base = os.path.join(self.svg_base, class_name)
            class_sketch_names = os.listdir(class_base)
            class_sketch_names.sort()
            class_sketch_names = [class_name + '/' + item for item in class_sketch_names]
            sketch_names += class_sketch_names
        self.sketch_names = sketch_names

    def __len__(self):
        return len(self.sketch_names)

    def __getitem__(self, item):
        sketch_name = self.sketch_names[item]  # e.g., "airplane/1.svg"
        class_name, file_name = sketch_name.split('/')
        svg_path = os.path.join(self.svg_base, class_name, file_name)

        # Parse the original data into the desired input format of SketchVG
        image_sizes, path_list, total_segment_num = parse_svg(svg_path)
        # image_sizes: [width, height]
        # path_list: list of (N_strokes, 4, 2)

        sketch = SketchVG(path_list=path_list)
        return sketch, total_segment_num
