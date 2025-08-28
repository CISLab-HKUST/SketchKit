import os
import gdown
import traceback
import json
import numpy as np
from sketchkit.core import Sketch
from sketchkit.utils.general_utils import line_to_cubic

from sketchkit.core.sketch import Sketch, Path, Stroke, Vertex, Point
from sketchkit.utils.general_utils.md5sum import *
from sketchkit.utils.general_utils.md5sum_table import *
from sketchkit.utils.general_utils.extract_files import *
from sketchkit.core import Sketch

COLORS_BIRD = {'initial':np.array([45, 169, 145])/255., 'eye':np.array([243, 156, 18])/255., 'none':np.array([149, 165, 166])/255., 
			'beak':np.array([211, 84, 0])/255., 'body':np.array([41, 128, 185])/255., 'details':np.array([171, 190, 191])/255.,
			'head':np.array([192, 57, 43])/255., 'legs':np.array([142, 68, 173])/255., 'mouth':np.array([39, 174, 96])/255., 
			'tail':np.array([69, 85, 101])/255., 'wings':np.array([127, 140, 141])/255.}

COLORS_CREATURE = {'initial':np.array([45, 169, 145])/255., 'eye':np.array([243, 156, 18])/255., 'none':np.array([149, 165, 166])/255., 
			'arms':np.array([211, 84, 0])/255., 'beak':np.array([41, 128, 185])/255., 'mouth':np.array([54, 153, 219])/255.,
			'body':np.array([192, 57, 43])/255., 'ears':np.array([142, 68, 173])/255., 'feet':np.array([39, 174, 96])/255., 
			'fin':np.array([69, 85, 101])/255., 'hair':np.array([127, 140, 141])/255., 'hands':np.array([45, 63, 81])/255.,
			'head':np.array([241, 197, 17])/255., 'horns':np.array([51, 205, 117])/255., 'legs':np.array([232, 135, 50])/255., 
			'nose':np.array([233, 90, 75])/255., 'paws':np.array([160, 98, 186])/255., 'tail':np.array([58, 78, 99])/255., 
			'wings':np.array([198, 203, 207])/255., 'details':np.array([171, 190, 191])/255.}

def download_creative_sketch(
    output_folder, 
    remove_sourcefile: bool = True,
    expected_md5: str | None = None,
    gdrive_id: str = "1Dstn3Cv2yAnxt1JXC0zPa1VPVDSz9gc-",
    filename: str = "creative_sketch.zip",
):
    """ Download dataset and extract it.
    Args:
        output_folder: The folder to download the dataset.
        remove_sourcefile: Whether to remove the source file after extraction.
        expected_md5: The expected MD5 checksum of the downloaded file.
        gdrive_id: The Google Drive ID of the dataset.
        filename: The name of the downloaded file.
    
    Returns:
        None
    """

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

    try:
        extract_files(
            file_name=os.path.join(output_folder, filename),
            output_folder=os.path.join(output_folder, "creative_sketches"),
            remove_sourcefile=remove_sourcefile,
        )
        # extract_files(file_name=os.path.join(output_folder, "sketches_png.zip"),
        #               output_folder=os.path.join(output_folder, "tu_berlin"),
        #               remove_sourcefile=remove_sourcefile)
    except Exception as e:
        print(e)
        return

    print("Creative Sketch is extracted!")



class CreativeSketch:
    """ The Creative Sketch dataset (https://songweige.github.io/projects/creative_sketech_generation/home.html).
    Args:
        json_base: The directory of the raw data in Creative Sketch dataset
        sketch_label: The label of the sketch, "bird" or "creature"
    """

    def __init__(
        self,
        data_base: str = ".cache/sketchkit/datasets",
        file_id: str = "1Dstn3Cv2yAnxt1JXC0zPa1VPVDSz9gc-",
        archive_name: str = "creative_sketch.zip",
        md5: str | None = None,
        remove_sourcefile: bool = True,
        split: str = "all", # "bird" or "creature" or "all"
        only_good: bool = True,
    ):
        """ Initialize the Creative Sketch dataset.
        Args:
            data_base: The folder to download the dataset.
            file_id: The Google Drive ID of the dataset.
            archive_name: The name of the downloaded file.
            md5: The expected MD5 checksum of the downloaded file.
            remove_sourcefile: Whether to remove the source file after extraction.
            split: Creative Sketch dataset has "bird" and "creature" two different subdatasets, use "all" to load both.
            only_good: Whether to only use the good samples.
        """

        data_root = os.path.join(os.environ.get("HOME"), data_base)
        os.makedirs(data_root, exist_ok=True)

        print("Downloading {} Database...".format(file_id))
        if not os.path.exists(os.path.join(data_root, "creative_sketches")):
            download_creative_sketch(
                output_folder=data_root,
                remove_sourcefile=remove_sourcefile,
                expected_md5=md5,
                gdrive_id=file_id,
                filename=archive_name,
            )
        else:
            print("Creative Sketch Database already exists.")

        bird_json_path = os.path.join(data_root, "creative_sketches", "raw_data_clean/creative_birds_json.txt")
        creature_json_path = os.path.join(data_root, "creative_sketches", "raw_data_clean/creative_creatures_json.txt")
        # self.json_base = json.loads(open(os.path.join(data_root, "creative_sketches")).read())
        print("Loading Creative Sketch Database...")
        if split == "bird":
            bird_json_base = json.loads(open(bird_json_path).read())
            self.json_base = [sketch for sketch in bird_json_base if sketch["good_sample"]] if only_good else bird_json_base
            self.sketch_label = ["bird"] * len(self.json_base)
        elif split == "creature":
            creature_json_base = json.loads(open(creature_json_path).read())
            self.json_base = [sketch for sketch in creature_json_base if sketch["good_sample"]] if only_good else creature_json_base
            self.sketch_label = ["creature"] * len(self.json_base)
        elif split == "all":
            bird_json_base = json.loads(open(bird_json_path).read())
            bird_json_base = [sketch for sketch in bird_json_base if sketch["good_sample"]] if only_good else bird_json_base
            creature_json_base = json.loads(open(creature_json_path).read())
            creature_json_base = [sketch for sketch in creature_json_base if sketch["good_sample"]] if only_good else creature_json_base
            self.json_base = bird_json_base + creature_json_base
            self.sketch_label = ["bird"] * len(bird_json_base) + ["creature"] * len(creature_json_base)
        else:
            raise Exception('Unknown split:', split)
        print("Done!")

    def __len__(self):
        return len(self.json_base)

    def __getitem__(self, item):

        single_sketch_json = self.json_base[item]  # (N_point, 3)
        label = self.sketch_label[item]

        if label == "bird":
            colors = COLORS_BIRD
        elif label == "creature":
            colors = COLORS_CREATURE
        else:
            raise Exception('Unknown label:', label)

        path_list = []
        for j, step in enumerate(single_sketch_json['all_strokes']):
            stroke_label = single_sketch_json["partsUsed"][j]
            stroke_color = colors[stroke_label]

            strokes = []
            for stroke in step:
                if len(stroke) == 0: # skip the empty stroke
                    continue
                for seg in stroke:
                    p_start = [seg[0], seg[1]]
                    p_end = [seg[2], seg[3]]
                    line = np.stack([p_start, p_end], axis=0)
                    cubic = line_to_cubic(line)
                    v_start = Vertex(p_start[0], p_start[1], color=stroke_color)
                    v_end = Vertex(p_end[0], p_end[1], color=stroke_color)
                    v_ctrl1 = Point(cubic[1][0], cubic[1][1])
                    v_ctrl2 = Point(cubic[2][0], cubic[2][1])
                    strokes.append(Stroke(v_start, v_end, v_ctrl1, v_ctrl2))
            path_list.append(Path(strokes))
        # path_list: list of (N_strokes, 4, 2)

        sketch = Sketch(paths=path_list)
        return sketch

