import os
import zipfile
import requests
import ndjson
import numpy as np
from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.general_utils import line_to_cubic
from collections import defaultdict
def download_sketchx_pris(output_folder=".cache/sketchkit/datasets", remove_zip=True):
    """Download SketchX-PRIS Dataset from GitHub and unzip it."""
    url = "https://github.com/KeLi-SketchX/SketchX-PRIS-Dataset/archive/refs/heads/master.zip"
    os.makedirs(output_folder, exist_ok=True)
    zip_path = os.path.join(output_folder, "SketchX-PRIS.zip")

    if not os.path.exists(zip_path):
        print("Downloading SketchX-PRIS Dataset...")
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        with open(zip_path, 'wb') as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    percent = downloaded / total_size * 100 if total_size > 0 else 0
                    print(f"\rDownloading... {percent:.1f}%", end="", flush=True)
        print("\nDownload finished.")
    else:
        print("Zip file already exists, skipping download.")

    # Extract zip
    extract_folder = os.path.join(output_folder, "SketchX-PRIS-Dataset")
    if not os.path.exists(extract_folder):
        print("Extracting...")
        print(zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
        extracted_subfolder = os.path.join(output_folder, "SketchX-PRIS-Dataset-main")
        if os.path.exists(extracted_subfolder):
            os.rename(extracted_subfolder, extract_folder)
        print("Extraction finished.")
    else:
        print("Dataset already extracted.")

    if remove_zip:
        os.remove(zip_path)
        print("Removed zip file.")

    return extract_folder
    
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

class SketchXPRIS:
    """ The SketchX-PRIS-Dataset (https://github.com/KeLi-SketchX/SketchX-PRIS-Dataset),
        which contains vector sketches across 25 categories.
        Each vector sketch is represented with polylines.

        Args:
            category: A category name in SketchX-PRIS-Dataset.
            data_root: The directory of storing the dataset, which is automatically downloaded if not existed.

        Attributes:
            stroke_data: A list of sketch, each in shape (N_point, 4). Each point is in stroke-4 format (dx, dy, pen_state, group_id).
                Pen state is 0 (drawing) or 1 (lifting).
                group_id is the Group IDS in each category

        Example:
            >>> my_data = SketchXPRIS(category="airplane")
            >>> print(len(my_data))
                800
            >>> sketch, sketches_by_groupid = dataset[1]
            >>> print(sketch)   
                <sketchkit.core.sketch.Sketch object at 0x7f178d598110>
            >>> print(sketches_by_groupid)
                [<sketchkit.core.sketch.Sketch object at 0x7f178f4f5810>, <sketchkit.core.sketch.Sketch object at 0x7f178f4f7c50>, <sketchkit.core.sketch.Sketch object at 0x7f178f503750>]
            >>> print(sketches_by_groupid[0].group_name)  
                body
    """
    def __init__(self, category="airplane", data_root=".cache/sketchkit/datasets"):
        self.data_root = data_root
        self.dataset_folder = os.path.join(data_root, "SketchX-PRIS-Dataset-master", "Perceptual Grouping")
        self.group_folder = os.path.join(data_root, "SketchX-PRIS-Dataset-master", "Group ID")
        if not os.path.exists(self.dataset_folder):
            download_sketchx_pris(data_root)
        
        filename = self.dataset_folder + "/" + category + ".ndjson"
        with open(filename, "r") as fp:
            self.data = ndjson.load(fp)
        self.category = category

        txt_path = os.path.join(self.group_folder, f"{self.category}.txt")
        self.group_id_to_name = {}
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            self.group_id_to_name = {i: name for i, name in enumerate(lines)}
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sketches = self.data[0]
        stroke_data = sketches["train_data"][idx]  # fallback to whole item
        path_list,total_segment_num = parse_stroke3_format(np.array(stroke_data))

        paths = []
        for p in path_list:
            curves = []
            i = 0
            for seg in p:
                p0 = Vertex(seg[0][0], seg[0][1])
                p3 = Vertex(seg[3][0], seg[3][1])
                c1 = Point(seg[1][0], seg[1][1])
                c2 = Point(seg[2][0], seg[2][1])
                curves.append(Curve(p0, p3, c1, c2))
                i += 1
            paths.append(Path(curves=curves))

        sketch_obj = Sketch(paths=paths)
        groupid_to_data = defaultdict(list)
        for s in stroke_data:
            groupid_to_data[int(s[-1])].append(s)
        sketches_by_groupid = []
        for gid, strokes in groupid_to_data.items():
            if len(strokes) == 0:
                continue
            strokes = np.array(strokes)
            path_list, _ = parse_stroke3_format(strokes)
            paths = []
            for p in path_list:
                curves = []
                for seg in p:
                    p0 = Vertex(seg[0][0], seg[0][1])
                    p3 = Vertex(seg[3][0], seg[3][1])
                    c1 = Point(seg[1][0], seg[1][1])
                    c2 = Point(seg[2][0], seg[2][1])
                    curves.append(Curve(p0, p3, c1, c2))
                paths.append(Path(curves=curves))
            sketch_obj_gid = Sketch(paths=paths)


            group_name = self.group_id_to_name.get(gid, f"unknown_{gid}")
            setattr(sketch_obj_gid, "group_name", group_name)
            sketches_by_groupid.append(sketch_obj_gid)
        return sketch_obj, sketches_by_groupid
       
