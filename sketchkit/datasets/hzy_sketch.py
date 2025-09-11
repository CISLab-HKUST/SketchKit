import os
import shutil

import pandas
import json

from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import download_with_wget, dir_md5, CISLAB_CDN, extract_files
from pathlib import Path as PathLibPath
from sketchkit.core import Sketch

dataset_name = "hzy_sketch"
dataset_filename = "hzy_sketch-samples.zip"


def download_and_extract_data(
    output_folder, remove_sourcefile=True, cislab_source=True
):
    os.makedirs(output_folder, exist_ok=True)
    try:
        if cislab_source:
            url = (
                "https://cislab.hkust-gz.edu.cn/demos/aisketch/data/"
                + f"{dataset_filename}"
            )
        else:
            # TODO: Add another source
            url = (
                "https://cislab.hkust-gz.edu.cn/demos/aisketch/data/"
                + f"{dataset_filename}"
            )
        download_with_wget(
            url,
            os.path.join(output_folder, dataset_filename),
        )
    except Exception as e:
        print(e)
        return

    try:
        extract_files(
            file_path=os.path.join(output_folder, dataset_filename),
            output_dir=output_folder,
            remove_sourcefile=remove_sourcefile,
        )

    except Exception as e:
        print(e)
        return


def load_hzy_sketch_json(drawing_path):
    """
    Load a vector sketch from a .json file and convert to SketchVG format.
    Args:
        drawing_path (str): Path to the .json file.
    Returns:
        Sketch: A SketchVG object representing the sketch.
    """
    with open(drawing_path, "rb") as fp:
        txt = fp.read().decode("utf-8")
    strokes = json.loads(txt)

    paths = []

    for stroke in strokes:
        points = stroke["points"]  # [{"x":x,y:"y","w":width,"t":time_stamp}...]
        pressures = stroke.get("pressure", [None] * len(points))
        thickness = stroke.get("width", None)

        curves = []
        for i in range(len(points) - 1):
            x0, y0 = points[i]["x"], points[i]["y"]
            x1, y1 = points[i + 1]["x"], points[i + 1]["y"]
            thickness = points[i]["w"]

            p0 = pressures[i] if pressures else None
            p1 = pressures[i + 1] if pressures else None

            v_start = Vertex(x0, y0, pressure=p0, thickness=thickness)
            v_end = Vertex(x1, y1, pressure=p1, thickness=thickness)

            ctrl1 = Point(2 / 3 * x0 + 1 / 3 * x1, 2 / 3 * y0 + 1 / 3 * y1)
            ctrl2 = Point(1 / 3 * x0 + 2 / 3 * x1, 1 / 3 * y0 + 2 / 3 * y1)

            curve = Curve(v_start, v_end, ctrl1, ctrl2)
            curves.append(curve)

        if curves:
            path = Path(curves=curves)
            paths.append(path)

    return Sketch(paths=paths)


class hzySketch(SketchDataset):
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

    md5_sum = "bb0d0fdc6eefa2e2ab41f990cc8cd5f4"

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
        super().__init__(root, load_all=load_all, cislab_source=cislab_source)

    def _check_integrity(self) -> bool:
        """Check the integrity of the cached dataset using MD5 checksum.

        Returns:
            bool: True if the dataset integrity is verified, False otherwise.
        """

        print(f"Checking integrity of cached {self.__class__.__name__} dataset...")
        current_md5 = dir_md5(self.root)
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

        print("MD5 Checksum:", dir_md5(self.root))

    def _load_all(self):
        for _, row in self.items_metadata.iterrows():
            self.raw_data[row["id"]] = load_hzy_sketch_json(
                os.path.join(self.json_base, row["json"])
            )

    def _load_items_metadata(self):
        self.json_base = os.path.join(self.root, "json")
        if not os.path.isdir(self.json_base):
            raise FileNotFoundError(
                f"JSON directory not found at '{self.json_base}'. "
                "Please check the download or extraction process."
            )

        dest_dir = self.json_base
        if not os.path.isdir(dest_dir):
            print(f"Warning: Directory not found, skipping. Path: {dest_dir}")
            return
        items_metadata = []
        for id, filename in enumerate(sorted(os.listdir(dest_dir))):
            if filename.endswith(".json"):
                items_metadata.append({"id": id, "json": filename})
        self.items_metadata = pandas.DataFrame(items_metadata)
        self.raw_data: list[Sketch | None] = [None] * len(self.items_metadata)

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
        if self.raw_data[idx] is None:
            self.json_base = os.path.join(self.root, "json")
            self.raw_data[idx] = load_hzy_sketch_json(
                os.path.join(self.json_base, self.items_metadata.iloc[idx]["json"])
            )
        sketch = self.raw_data[idx]
        if sketch is None:
            raise RuntimeError(f"Failed to load sketch at index {idx}")
        return sketch


if __name__ == "__main__":
    dataset = hzySketch(load_all=True)
    print(dataset)
    sketch = dataset[1]
    print(sketch.curve_num, sketch.path_num)

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer

    renderer = CairoRenderer(1024, (1, 1, 1))
    raster_image = renderer.render(sketch)
    outpath = "test.png"
    if outpath is not None:
        raster_image_png = Image.fromarray(raster_image, "RGB")
        raster_image_png.save(outpath, "PNG")
