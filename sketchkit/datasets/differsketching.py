import os
import requests
import zipfile
from tqdm import tqdm
import json
from pathlib import Path as SysPath
from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point

def download_differsketching(output_folder):
    """
    Download the DifferSketching dataset from the given URL and extract it.

    Args:
        output_folder (str): The directory to save the downloaded dataset.
    """
    os.makedirs(output_folder, exist_ok=True)

    url = "https://drive.google.com/uc?id=1A_3RVc8Y4YdI7nhyM7tb-q7dQw4zTcCO&export=download"
    zip_path = os.path.join(output_folder, "differsketching.zip")
    extracted_folder = os.path.join(output_folder, "differsketching")

    if os.path.exists(extracted_folder):
        print("Differsketching already exists and is extracted. Skipping download and extraction.")
        return extracted_folder

    if not os.path.exists(zip_path):
        print("Downloading Differsketching dataset...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(zip_path, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    else:
        print("zip already exists. Skipping download.")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_folder)

    print("Extracting done.！")
    return extracted_folder


def load_differsketching_json(json_path):
    """
    Load a vector sketch from a .json file and convert to SketchVG format.
    Args:
        json_path (str): Path to the .json file.
    Returns:
        Sketch: A SketchVG object representing the sketch.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    strokes = data["strokes"]
    paths = []

    for stroke in strokes:
        points = stroke["path"]               # [[x,y], [x,y], ...]
        pressures = stroke.get("pressure", [None] * len(points))
        thickness = stroke.get("width", None)

        curves = []
        for i in range(len(points) - 1):
            x0, y0 = points[i]
            x1, y1 = points[i + 1]

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

def convert_differsketching(base_folder):
    """
    Extract .npy files from all categories and convert them to SketchVG format.

    Args:
        base_folder (str): The base folder containing all categories with .json files.
    """
    all_sketches = []
    base = SysPath(base_folder)

    for json_dir in base.rglob("*_json"):
        for json_file in json_dir.glob("*.json"):
            try:
                sketch = load_differsketching_json(json_file)
                print(f"Converted: {json_file} → {sketch.path_num} paths, {sketch.curve_num} curves")
                all_sketches.append((json_file, sketch))
            except Exception as e:
                print(f"Failed to convert {json_file}: {e}")

    return all_sketches

if __name__ == "__main__":
    output_folder = "data/differsketching"

    print("Starting dataset download and extraction...")
    dataset_path = download_differsketching(output_folder)

    print("Starting conversion: JSON → Sketch...")
    sketches = convert_differsketching(dataset_path)

    print(f"All tasks completed successfully. {len(sketches)} sketches in total.")
