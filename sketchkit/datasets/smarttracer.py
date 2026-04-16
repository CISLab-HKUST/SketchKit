import os
import shutil

import pandas
import json

from sketchkit.core.sketch import Sketch, Path, Curve, Point
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import download_with_wget, dir_md5, CISLAB_CDN, extract_files
from sketchkit.core import Sketch

dataset_filename = "sketch.zip"

DEFAULT_CANVAS_HEIGHT= 1024
DEFAULT_CANVAS_WIDTH = 1024

def download_and_extract_data(
    output_folder, remove_sourcefile=True, cislab_source=True
):
    os.makedirs(output_folder, exist_ok=True)
    url = f"{CISLAB_CDN}/datasets/SmartTracer/{dataset_filename}"
    try:
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


def load_smarttracer_sketch_json(drawing_path):
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

            v_start = Point(x0, y0, pressure=p0, thickness=thickness)
            v_end = Point(x1, y1, pressure=p1, thickness=thickness)

            ctrl1 = Point(2 / 3 * x0 + 1 / 3 * x1, 2 / 3 * y0 + 1 / 3 * y1)
            ctrl2 = Point(1 / 3 * x0 + 2 / 3 * x1, 1 / 3 * y0 + 2 / 3 * y1)

            curve = Curve(v_start, v_end, ctrl1, ctrl2)
            curves.append(curve)

        if curves:
            path = Path(curves=curves)
            paths.append(path)

    return Sketch(paths=paths, height=DEFAULT_CANVAS_HEIGHT, width=DEFAULT_CANVAS_WIDTH)


class SmartTracer(SketchDataset):
    """The SmartTracer dataset contains drawing process for high-quality anime line arts in json files.

    Attributes:
        md5_sum (str): MD5 checksum for dataset integrity verification.

    """
    md5_sum = "473c51ae0259f1435994a9cea090b5e7"

    # def __init__(
    #     self,
    #     root: str | PathLibPath | None = None,
    #     load_all: bool = False,
    #     cislab_source: bool = False,
    # ):
    #     """Initializes the SmartTracer dataset loader.

    #     Checks if the dataset exists locally. If not, it triggers the download
    #     and extraction process. It then scans the SVG directories to populate a
    #     list of all available sketch file paths.

    #     Args:
    #         data_base: The base directory for storing all datasets.
    #             Defaults to '~/.sketchkit/datasets'.
    #     """
    #     super().__init__(root, load_all=load_all, cislab_source=cislab_source)

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
            self.raw_data[row["id"]] = load_smarttracer_sketch_json(
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

    def _get_single(self, idx: int) -> Sketch:
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
            self.raw_data[idx] = load_smarttracer_sketch_json(
                os.path.join(self.json_base, self.items_metadata.iloc[idx]["json"])
            )
        sketch = self.raw_data[idx]
        if sketch is None:
            raise RuntimeError(f"Failed to load sketch at index {idx}")
        return sketch


if __name__ == "__main__":
    dataset = SmartTracer()
    print(dataset)
    sketch = dataset[1]
    print(sketch.curve_num, sketch.path_num)

    from sketchkit.renderer import CairoRenderer

    renderer = CairoRenderer()
    raster_image = renderer.render(sketch)
    raster_image.save("test.png")
