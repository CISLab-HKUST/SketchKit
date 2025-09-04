import os
import wget
import json
#import xml.etree.ElementTree as ET
#from svg.path import parse_path, path

from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import download_with_tqdm, dir_md5, CISLAB_CDN, extract_files
from pathlib import Path as PathLibPath
from sketchkit.core import Sketch

dataset_name="hzy_sketch"
dataset_filename="hzy_sketch-samples.zip"
def download_and_extract_data(output_folder, remove_sourcefile=True,cislab_source=True):
    print("Downloading started......")

    os.makedirs(output_folder, exist_ok=True)
    try:
        if cislab_source:
            url = "https://cislab.hkust-gz.edu.cn/demos/aisketch/data/"+f"{dataset_filename}"
        else:
            #TODO: Add another source
            url = "https://cislab.hkust-gz.edu.cn/demos/aisketch/data/"+f"{dataset_filename}"
        wget.download(
            url = url,
            out=os.path.join(output_folder, dataset_filename),
        )
    except Exception as e:
        print(e)
        return

    print("hzy is downloaded!")
    '''
    assert (
        calculate_md5(os.path.join(output_folder, "sketches_svg.zip"))
        == MD5SUM_TABLE["sketches_svg.zip"]
    ), "sketches_svg.zip md5 checksum error"


    print("Md5 checksum passed!")
    '''
    print("Extracting started......")

    try:
        extract_files(
            file_path=os.path.join(output_folder, dataset_filename),
            output_dir=os.path.join(output_folder, dataset_name),
            remove_sourcefile=remove_sourcefile,
        )

    except Exception as e:
        print(e)
        return

    print(f"{dataset_name} is extracted!")

def load_hzy_sketch_json(drawing_path):
    """
    Load a vector sketch from a .json file and convert to SketchVG format.
    Args:
        drawing_path (str): Path to the .json file.
    Returns:
        Sketch: A SketchVG object representing the sketch.
    """
    with open(drawing_path, "rb") as fp:
        txt=fp.read().decode('utf-8')
    strokes=json.loads(txt)
            
    paths = []

    for stroke in strokes:
        points = stroke["points"]               # [{"x":x,y:"y","w":width,"t":time_stamp}...]
        pressures = stroke.get("pressure", [None] * len(points))
        thickness = stroke.get("width", None)

        curves = []
        for i in range(len(points) - 1):
            x0, y0 = points[i]["x"],points[i]["y"]
            x1, y1 = points[i + 1]["x"],points[i + 1]["y"]
            thickness = points[i]["w"]

            p0 = pressures[i] if pressures else None
            p1 = pressures[i + 1] if pressures else None

            v_start = Vertex(x0, y0, pressure=p0, thickness=thickness)
            v_end   = Vertex(x1, y1, pressure=p1, thickness=thickness)


            ctrl1 = Point(2/3*x0 + 1/3*x1, 2/3*y0 + 1/3*y1)
            ctrl2 = Point(1/3*x0 + 2/3*x1, 1/3*y0 + 2/3*y1)

            curve = Curve(v_start, v_end, ctrl1, ctrl2)
            curves.append(curve)

        if curves:
            path = Path(curves=curves)
            paths.append(path)

    return Sketch(paths=paths)

class hzy_sketch(SketchDataset):
    """The hzy dataset,
    which contains drawing process for high-quality anime line arts.

    Args:
        data_base: The directory of storing the dataset, which is automatically downloaded if not existed.
    """
    """Loads the hzy_sketch dataset from json files.

    This class handles the automatic download, extraction, and parsing of the
    hzy_sketch dataset. It indexes all json sketches and provides an
    interface to access them as `Sketch` objects. Accessing an item
    (e.g., `my_data[0]`) returns a single `Sketch` object.


    
    Attributes:
        md5_sum (str): MD5 checksum for dataset integrity verification.
        
    Example:
        >>> my_data = hzy_sketch()
        >>> print(len(my_data))
        182
        >>> print(my_data[0])
       <sketchkit.core.sketch.Sketch object at 0x000001AB5033ACE0>
    """

    def __init__(
        self,
        root: str | PathLibPath | None = None,
        load_all: bool = False,
        cislab_source: bool = False,
    ):
        """Initializes the hzy dataset loader.

        Checks if the dataset exists locally. If not, it triggers the download
        and extraction process. It then scans the SVG directories to populate a
        list of all available sketch file paths.

        Args:
            data_base: The base directory for storing all datasets.
                Defaults to '~/.cache/sketchkit/datasets'.
        """
        self.md5_sum="bb0d0fdc6eefa2e2ab41f990cc8cd5f4"
        super().__init__(root, load_all=load_all, cislab_source=cislab_source)
        #self.data_root = os.path.join(self.root, dataset_name)
        #self.root = os.path.join(data_base, dataset_name)
        
        

    def _check_integrity(self) -> bool:
        """Check the integrity of the cached dataset using MD5 checksum.

        Returns:
            bool: True if the dataset integrity is verified, False otherwise.
        """
        
        print(f"Checking integrity of cached {self.__class__.__name__} dataset...")
        #return True
        current_md5 = dir_md5(self.root)
        print(current_md5,self.root)
        return current_md5 == self.md5_sum

    def _download(self):
        """Download the QuickDraw dataset.

        Downloads the categories.txt file and all category .npz files from either
        the original Google storage or CISLAB CDN mirror. Creates the root directory
        if it doesn't exist.

        Raises:
            Exception: If download fails for any file.
        """

            
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

        download_and_extract_data(self.root)
        
        #print("MD5 Checksum:", dir_md5(self.root))
    def _load_all(self):

        self.json_base = os.path.join(self.root, dataset_name,"json")
        if not os.path.isdir(self.json_base):
            raise FileNotFoundError(
                f"JSON directory not found at '{self.json_base}'. "
                "Please check the download or extraction process."
            )
        json_list=[]
        

        dest_dir = self.json_base
        if not os.path.isdir(dest_dir):
            print(f"Warning: Directory not found, skipping. Path: {dest_dir}")
            return
        for filename in sorted(os.listdir(dest_dir)):
            if filename.endswith(".json"):
                json_list.append(os.path.join(dest_dir, filename))
        self.raw_data = [None]*len(json_list)
        for i in range(len(json_list)):
            self.raw_data[i]=json_list[i]
        print('json_list',len(json_list))
    def _load_items_metadata(self):
        pass

    def __len__(self) -> int:
        return len(self.raw_data)
        
    def __getitem__(self, idx: int) -> Sketch:
        """Retrieves a sketch by its index.

        Args:
            idx: The index of the sketch to retrieve.

        Returns:
            The parsed sketch as a `Sketch` object.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        json_path = self.raw_data[idx]
        return load_hzy_sketch_json(json_path)

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        data_base = os.path.join(os.path.expanduser("~"), ".cache/sketchkit/datasets")
        data = hzy_sketch(data_base)
        print(f"Total sketches: {len(data)}")
        sketch = data[0]
        print(sketch)
        print(sketch.curve_num, sketch.path_num)
        print("Loaded first sketch successfully.")