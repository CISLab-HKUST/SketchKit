import os
import json
import shutil
import pandas as pd
import traceback
from tqdm import tqdm
from pathlib import Path as PathLibPath

from sketchkit.core.sketch import Sketch, Path, Curve, Point
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.file import download_with_wget, dir_md5, CISLAB_CDN, extract_files

def parse_hex_color(hex_str: str) -> tuple[float, float, float]:
    """Parses a hex color string (e.g., '#000000') into an RGB tuple (0.0-1.0)."""
    if not hex_str or not isinstance(hex_str, str) or not hex_str.startswith("#"):
        return (0.0, 0.0, 0.0)  # Default to black
    hex_str = hex_str.lstrip("#")
    try:
        return tuple(int(hex_str[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return (0.0, 0.0, 0.0)

class TracingVsFreehand(SketchDataset):
    """
    Tracing-vs-Freehand dataset loader.
    
    This loader reads from the raw JSON files (tracings.json, drawings.json, etc.)
    instead of the SVG files to access rich attributes like pressure and opacity.
    
    Attributes:
        md5_sum (str): MD5 checksum for dataset integrity verification.
    """

    md5_sum = "d069ddc535281d50e271ee8bcbcd091e"
    
    # Official canvas size stated in the dataset documentation
    CANVAS_WIDTH = 800
    CANVAS_HEIGHT = 800

    # Metadata columns for indexing
    metadata = ["id", "json_filename", "prompt_key", "participant_id", "sketch_type"]

    def __init__(
        self,
        root: PathLibPath | str | None = None,
        load_all: bool = False,
        cislab_source: bool = True,
    ):
        super().__init__(root, load_all=load_all, cislab_source=cislab_source)
        # Cache for holding loaded JSON content to avoid repeated I/O
        self._json_cache = {}

    def _check_integrity(self) -> bool:
        """Checks if the dataset is present and uncorrupted."""
        print(f"Checking integrity of cached {self.__class__.__name__} dataset...")
        current_md5 = dir_md5(self.root)
        return current_md5 == self.md5_sum

    def _download(self):
        """Downloads and extracts the dataset."""
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)

        zip_path = os.path.join(self.root, "sketch.zip")

        if self.cislab_source:
            print("Using CISLAB CDN for download...")
            url = (
                f"{CISLAB_CDN}/datasets/{self.__class__.__name__}/" + "sketch.zip"
            )
            download_with_wget(
                url,
                file_path=zip_path,
                pg_bar=True,
            )
        else:
            try:
                print("Downloading from official hosting...")
                url = "https://cislab.hkust-gz.edu.cn/projects/tracing-vs-freehand/data/sketch.zip"
                download_with_wget(
                    url,
                    file_path=zip_path,
                    pg_bar=True,
                )
                print("\nDownload finished!")
            except Exception as e:
                traceback.print_exc()
                print("Download failed.")
                raise e

        try:
            extract_files(
                file_path=zip_path,
                output_dir=self.root,
                remove_sourcefile=True,
            )
        except Exception as e:
            raise e

        print("MD5 Checksum:", dir_md5(self.root))

    def _load_items_metadata(self):
        """
        Scans the main JSON files to build a metadata index.
        
        The dataset is organized into three large JSON files:
        - tracings.json
        - drawings.json
        - drawings_registered.json
        """
        metadata_cache_path = os.path.join(self.root, ".metadata_json.parquet")
        target_files = ["tracings.json", "drawings.json", "drawings_registered.json"]
        
        # Load from cache if available
        if os.path.exists(metadata_cache_path):
            self.items_metadata = pd.read_parquet(metadata_cache_path)
            # Verify columns match
            if set(self.items_metadata.columns) == set(self.metadata):
                self.raw_data = [None] * len(self.items_metadata)
                return

        print("Scanning JSON files to build metadata index...")
        metadata_list = []
        cnt = 0
        
        data_dir = os.path.join(self.root, "data")
        if not os.path.isdir(data_dir):
             raise FileNotFoundError(f"Data directory not found at {data_dir}")

        for json_file in target_files:
            full_path = os.path.join(data_dir, json_file)
            if not os.path.exists(full_path):
                print(f"Warning: {json_file} not found, skipping.")
                continue
            
            print(f"Indexing {json_file}...")
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    # We only read the keys to build the index
                    data = json.load(f)
                    
                # Structure: { "image_filename.png": { "participant_id": [strokes...], ... }, ... }
                for prompt_key, participants in data.items():
                    for participant_id in participants.keys():
                        metadata_list.append({
                            "id": cnt,
                            "json_filename": json_file,
                            "prompt_key": prompt_key,
                            "participant_id": participant_id,
                            "sketch_type": json_file.replace(".json", "")
                        })
                        cnt += 1
            except json.JSONDecodeError as e:
                print(f"Error parsing {json_file}: {e}")

        if not metadata_list:
            raise RuntimeError("No valid data found in JSON files.")

        self.items_metadata = pd.DataFrame(metadata_list)
        self.items_metadata.to_parquet(metadata_cache_path)
        self.raw_data = [None] * len(self.items_metadata)

    def _get_json_content(self, json_filename: str):
        """
        Helper to load and cache the content of a large JSON file.
        This prevents re-reading the file for every single sketch.
        """
        if json_filename not in self._json_cache:
            path = os.path.join(self.root, "data", json_filename)
            print(f"Loading {json_filename} into memory...")
            with open(path, 'r', encoding='utf-8') as f:
                self._json_cache[json_filename] = json.load(f)
        return self._json_cache[json_filename]

    def _load_all(self):
        """Loads all sketches into memory."""
        print("Loading all sketches into memory...")
        for idx in tqdm(range(len(self.items_metadata)), desc="Loading Data"):
            self._get_single(idx)

    def _get_single(self, idx: int) -> Sketch:
        """
        Retrieves a single sketch by its index, parsing raw JSON data.
        """
        if not 0 <= idx < len(self):
            raise IndexError("Index out of range")
        
        if self.raw_data[idx] is not None:
            return self.raw_data[idx]

        # Retrieve metadata
        row = self.items_metadata.iloc[idx]
        json_filename = row["json_filename"]
        prompt_key = row["prompt_key"]
        p_id = row["participant_id"]

        # Get full dataset content (cached)
        full_data = self._get_json_content(json_filename)
        
        # Extract the specific sketch data (list of strokes)
        if prompt_key not in full_data or p_id not in full_data[prompt_key]:
            # Fallback for data inconsistency
            return Sketch(height=self.CANVAS_HEIGHT, width=self.CANVAS_WIDTH, paths=[])

        strokes_data = full_data[prompt_key][p_id]
        paths = []

        for stroke_entry in strokes_data:
            # 1. Parse Path String (Format: "t, x, y, t, x, y ...")
            path_str = stroke_entry.get("path", "")
            if not path_str:
                continue
            
            try:
                path_vals = [float(v) for v in path_str.split(",")]
            except ValueError:
                continue

            if len(path_vals) % 3 != 0:
                continue 
            
            # Extract x and y, ignore timestamp (t)
            # data structure: [t0, x0, y0, t1, x1, y1, ...]
            xs = path_vals[1::3]
            ys = path_vals[2::3]
            num_points = len(xs)

            if num_points < 2:
                continue

            # 2. Parse Pressure String (Format: "p0, p1, ...")
            pres_str = stroke_entry.get("pressure", "")
            pressures = []
            if pres_str:
                try:
                    pressures = [float(v) for v in pres_str.split(",")]
                except ValueError:
                    pressures = []
            
            # Pad pressures if length mismatch (or default to None)
            if len(pressures) < num_points:
                pressures.extend([None] * (num_points - len(pressures)))

            # 3. Parse Attributes
            # Width is defined per stroke in this dataset (integer)
            stroke_width = float(stroke_entry.get("width", 1.0))
            
            # Color is a hex string
            color_hex = stroke_entry.get("color", "#000000")
            color_rgb = parse_hex_color(color_hex)
            
            # Opacity is a float (0.0 - 1.0)
            opacity = float(stroke_entry.get("opacity", 1.0))

            # 4. Construct Curves (Linear segments between points)
            curves = []
            for i in range(num_points - 1):
                x0, y0 = xs[i], ys[i]
                x1, y1 = xs[i+1], ys[i+1]
                
                # Pressure for start and end endpoints
                p0 = pressures[i]
                p1 = pressures[i+1]

                v_start = Point(
                    x=x0, y=y0, 
                    pressure=p0, 
                    thickness=stroke_width, 
                    color=color_rgb, 
                    opacity=opacity
                )
                v_end = Point(
                    x=x1, y=y1, 
                    pressure=p1, 
                    thickness=stroke_width, 
                    color=color_rgb, 
                    opacity=opacity
                )
                
                # Use linear interpolation for control points (straight lines)
                # Since the raw data is dense sampling points, straight lines are appropriate.
                ctrl1 = Point(2/3 * x0 + 1/3 * x1, 2/3 * y0 + 1/3 * y1)
                ctrl2 = Point(1/3 * x0 + 2/3 * x1, 1/3 * y0 + 2/3 * y1)
                
                curves.append(Curve(v_start, v_end, ctrl1, ctrl2))

            if curves:
                paths.append(Path(curves=curves))

        # Create Sketch object with fixed official dimensions
        sketch = Sketch(
            paths=paths, 
            height=self.CANVAS_HEIGHT, 
            width=self.CANVAS_WIDTH
        )
        
        self.raw_data[idx] = sketch
        return sketch

    def extra_repr(self) -> str:
        """Returns a string with extra information about the dataset."""
        if not hasattr(self, "items_metadata") or self.items_metadata.empty:
            return "Metadata not loaded."
        counts = self.items_metadata["sketch_type"].value_counts()
        return "\n".join([f"{k}: {v}" for k, v in counts.items()])
