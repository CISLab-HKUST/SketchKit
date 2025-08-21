import os
import wget
import numpy as np

from sketchkit.core.sketch import SketchVG
from sketchkit.utils.general_utils import line_to_cubic


def download_quickdraw(output_folder, category):
    os.makedirs(output_folder, exist_ok=True)

    category_txt = os.path.join(output_folder, "categories.txt")
    if not os.path.exists(category_txt):
        try:
            wget.download("https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/refs/heads/master/categories.txt",
                          out=category_txt)
        except Exception as e:
            print(e)
            return

    all_categories = []
    with open(category_txt, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            all_categories.append(line)

    if category not in all_categories:
        raise Exception("Category " + category + " is not in QuickDraw dataset.")

    npz_path = os.path.join(output_folder, category + ".npz")
    if not os.path.exists(npz_path):
        print("Downloading started......")
        try:
            wget.download("https://storage.googleapis.com/quickdraw_dataset/sketchrnn/" + category + ".npz",
                          out=npz_path)
        except Exception as e:
            print(e)
            return

        print("The category:", category, "in QuickDraw is downloaded!")


def parse_stroke3_format(stroke_data):
    """ Convert stroke-3 format in QuickDraw into our desired input format of SketchVG.

    Args:
        stroke_data: (N_point, 3), each point is in stroke-3 format (dx, dy, pen_state). Pen state is 0 (drawing) or 1 (lifting).

    Returns:
        path_list: path_list: list of (N_strokes, 4, 2)
    """
    dx = stroke_data[:, 0]  # (N_point)
    dy = stroke_data[:, 1]  # (N_point)
    pen_state = stroke_data[:, 2]  # (N_point)

    # Convert to absolute coordinates
    xs, ys = np.cumsum(dx), np.cumsum(dy)
    xs -= np.min(xs)
    ys -= np.min(ys)
    xy = np.stack([xs, ys], axis=1)  # (N_point, 2)

    path_list = []  # list of (N_strokes, 4, 2)
    stroke_list = []  # list of (4, 2)
    total_segment_num = 0
    for i in range(len(pen_state) - 1):
        if pen_state[i] == 0:
            p_start = xy[i]  # (2)
            p_end = xy[i + 1]  # (2)
            line = np.stack([p_start, p_end], axis=0)  # (2, 2)
            cubic = line_to_cubic(line)  # (4, 2)
            stroke_list.append(cubic)
            total_segment_num += 1
        else:
            path_list.append(np.stack(stroke_list, axis=0))
            stroke_list = []
    return path_list, total_segment_num


class QuickDraw:
    """ The QuickDraw dataset (https://quickdraw.withgoogle.com/data),
    which contains vector sketches across 345 categories.
    Each vector sketch is represented with polylines.

    Args:
        category: A category name in QuickDraw.
        split: "train", "valid", "test", or "all".
        data_base: The directory of storing the dataset, which is automatically downloaded if not existed.

    Attributes:
        stroke_data: A list of sketch, each in shape (N_point, 3). Each point is in stroke-3 format (dx, dy, pen_state).
            Pen state is 0 (drawing) or 1 (lifting).

    Example:
        >>> my_data = QuickDraw(category="cat", split="train")
        >>> print(len(my_data))
        70000
        >>> print(my_data[0])
        (<sketchkit.core.sketch.SketchVG object at 0x7555af6973a0>, 81)
    """

    def __init__(self, category="cat", split="train", data_base=".cache/sketchkit/datasets"):
        self.category = category
        self.split = split

        data_root = os.path.join(os.environ.get('HOME'), data_base)
        os.makedirs(data_root, exist_ok=True)

        # First, download the dataset if not existed.
        download_quickdraw(os.path.join(data_root, "quickdraw"), category)

        # Then, load the dataset to obtain data like class names and all the file names.
        data_filepath = os.path.join(data_root, "quickdraw", category + ".npz")
        data = np.load(data_filepath, encoding='latin1', allow_pickle=True)
        if split in ["train", "valid", "test"]:
            self.stroke_data = data[split]  # a list of (N_point, 3)
        elif split == "all":
            train_strokes = data['train']
            valid_strokes = data['valid']
            test_strokes = data['test']
            self.stroke_data = np.concatenate([train_strokes, valid_strokes, test_strokes])
        else:
            raise Exception('Unknown split:', split)

    def __len__(self):
        return len(self.stroke_data)

    def __getitem__(self, item):
        single_sketch_stroke = self.stroke_data[item]  # (N_point, 3)

        # Parse the original data into the desired input format of SketchVG
        path_list, total_segment_num = parse_stroke3_format(single_sketch_stroke)
        # path_list: list of (N_strokes, 4, 2)

        sketch = SketchVG(path_list=path_list)
        return sketch, total_segment_num
