"""
Script: fscoco.py

Description:
This script downloads the FS-COCO dataset, extracts its contents, and processes .npy files containing vector sketches. The .npy files are converted into the SketchVG format, which involves generating Bézier curves for each stroke in the sketch.

Main Functionality:
- Download and extract the FS-COCO dataset.
- Detect if the dataset or its extracted contents already exist to skip redundant steps.
- Load vector sketches from .npy files and convert them into SketchVG format using Bézier curve interpolation.
- Print progress and status messages during each step.

Usage:
- Run the script directly to download, extract, and process the dataset.
- Example:
    python fscoco.py

Dependencies:
- NumPy
- Requests
- TQDM
- SketchKit library

"""


import os
import numpy as np
import requests
import tarfile
from tqdm import tqdm
from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point

def download_fs_coco(output_folder):
    """
    Download the FS-COCO dataset from the given URL and extract it.

    Args:
        output_folder (str): The directory to save the downloaded dataset.
    """
    print("The download of FS-COCO has started......")
    os.makedirs(output_folder, exist_ok=True)

    url = "http://cvssp.org/data/fscoco/fscoco.tar.gz"
    tar_path = os.path.join(output_folder, "fscoco.tar.gz")
    extracted_folder = os.path.join(output_folder, "fscoco")

    # Check if the dataset is already extracted
    if os.path.exists(extracted_folder):
        print("FS-COCO dataset already exists and is extracted. Skipping download and extraction.")
        return

    # Check if the tar.gz file already exists
    if os.path.exists(tar_path):
        print("FS-COCO tar.gz file already exists. Skipping download.")
    else:
        print("Downloading FS-COCO dataset...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(tar_path, "wb") as f, tqdm(
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

    print("Extracting FS-COCO dataset...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(output_folder)

    # Remove the tar.gz file after extraction
    os.remove(tar_path)
    print("FS-COCO dataset is downloaded and extracted!")

def load_npy_vector_sketch(npy_path):
    """
    Load a vector sketch from a .npy file (N, 3) and convert to SketchVG format.
    Args:
        npy_path (str): Path to the .npy file.
    Returns:
        Sketch: A SketchVG object representing the sketch.
    """
    data = np.load(npy_path)  # shape: (N, 3)
    strokes = []
    current_stroke = []
    for point in data:
        x, y, flag = point
        current_stroke.append((x, y))
        if flag == 1:
            if len(current_stroke) > 1:
                for i in range(len(current_stroke) - 1):
                    start = current_stroke[i]
                    end = current_stroke[i + 1]
                    ctrl1 = (2/3*start[0] + 1/3*end[0], 2/3*start[1] + 1/3*end[1])
                    ctrl2 = (1/3*start[0] + 2/3*end[0], 1/3*start[1] + 2/3*end[1])
                    p_start = Vertex(start[0], start[1])
                    p_end = Vertex(end[0], end[1])
                    p_ctrl1 = Point(ctrl1[0], ctrl1[1])
                    p_ctrl2 = Point(ctrl2[0], ctrl2[1])
                    strokes.append(Curve(p_start, p_end, p_ctrl1, p_ctrl2))
            current_stroke = []
    if strokes:
        sketch = Sketch(paths=[Path(curves=strokes)])
    else:
        sketch = Sketch(paths=[])
    return sketch

def extract_and_convert_npy_files(base_folder):
    """
    Extract .npy files from subdirectories 1-100 and convert them to SketchVG format.

    Args:
        base_folder (str): The base folder containing subdirectories 1-100 with .npy files.
    """
    vector_sketches_folder = os.path.join(base_folder, "fscoco", "vector_sketches")

    for i in range(1, 101):
        subfolder = os.path.join(vector_sketches_folder, str(i))
        if not os.path.exists(subfolder):
            print(f"Subfolder {subfolder} does not exist. Skipping...")
            continue

        npy_files = [f for f in os.listdir(subfolder) if f.endswith(".npy")]
        for npy_file in npy_files:
            npy_path = os.path.join(subfolder, npy_file)
            try:
                sketch = load_npy_vector_sketch(npy_path)
                print(f"Converted {npy_path} to SketchVG format with {len(sketch.paths)} paths.")
            except Exception as e:
                print(f"Failed to convert {npy_path}: {e}")

if __name__ == "__main__":
    try:
        # Define the output folder for the dataset
        output_folder = "data/fs_coco"

        # Step 1: Download and extract the dataset
        print("Starting dataset download and extraction...")
        download_fs_coco(output_folder)

        # Step 2: Extract and convert .npy files to SketchVG format
        print("Starting conversion of .npy files to SketchVG format...")
        extract_and_convert_npy_files(output_folder)

        print("All tasks completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")