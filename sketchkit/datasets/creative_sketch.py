import os
import gdown
import traceback
import json
import numpy as np
from pathlib import Path as PathLibPath
import pandas as pd
from tqdm import tqdm

from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.dataset import SketchDataset
from sketchkit.utils.transform import line_to_cubic
from sketchkit.utils.file import *

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
    remove_sourcefile: bool = False,
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
        md5_val = file_md5(archive_path)
        assert md5_val == expected_md5, f"MD5 mismatch: {md5_val} != {expected_md5}"
        print("MD5 checksum passed!")
    else:
        print(file_md5(archive_path))

    print("Extracting archive...")

    try:
        extract_files(
            file_path=os.path.join(output_folder, filename),
            output_dir=os.path.join(output_folder, "creative_sketches"),
            remove_sourcefile=remove_sourcefile,
        )
    except Exception as e:
        print(e)
        return

    print("Creative Sketch is extracted!")



class CreativeSketch(SketchDataset):
    """The Creative Sketch dataset contains vector sketches of birds and creatures with detailed part annotations.

    Attributes:
        md5_sum (str): MD5 checksum for dataset integrity verification.

    References:
        - Original dataset: https://songweige.github.io/projects/creative_sketech_generation/home.html
    """

    # MD5 Sum of the dataset
    md5_sum = "84c57a01499321bd6080b1f76754d709" # 可以根据实际情况添加MD5校验和

    # Metadata columns for each item
    metadata = ["id", "category", "split", "is_good"]

    def __init__(
        self,
        root: str | PathLibPath | None = None,
        load_all: bool = False,
        cislab_source: bool = False,
    ):
        """Initialize the Creative Sketch dataset.

        Args:
            root (str | PathLibPath | None): Root directory to store/load the dataset.
                If None, uses the default dataset directory.
            load_all (bool): If True, loads all data into memory for faster access.
                If False, loads data on-demand. Defaults to False.
            cislab_source (bool): If True, downloads from CISLAB CDN mirror.
                If False, downloads from original Google storage. Defaults to False.
        """
        super().__init__(root, load_all=load_all, cislab_source=cislab_source)

    def _check_integrity(self) -> bool:
        """Check the integrity of the cached dataset.

        Returns:
            bool: True if the dataset integrity is verified, False otherwise.
        """
        print(f"Checking integrity of cached {self.__class__.__name__} dataset...")
        # zip文件md5检查
        # 检查zip文件是否存在
        if not os.path.exists(os.path.join(self.root, "creative_sketch.zip")):
            return False
        # 检查zip文件md5
        if file_md5(os.path.join(self.root, "creative_sketch.zip")) != self.md5_sum:
            return False
        # 检查数据集文件夹是否存在
        if not os.path.exists(os.path.join(self.root, "creative_sketches")):
            return False
        # 检查必要的JSON文件是否存在
        bird_json_path = os.path.join(self.root, "creative_sketches", "raw_data_clean/creative_birds_json.txt")
        creature_json_path = os.path.join(self.root, "creative_sketches", "raw_data_clean/creative_creatures_json.txt")
        return os.path.exists(bird_json_path) and os.path.exists(creature_json_path)

    def _download(self):
        """Download the Creative Sketch dataset.

        Downloads the dataset from Google Drive and extracts it to the root directory.

        Raises:
            Exception: If download fails for any reason.
        """
        # 复用现有的下载函数
        download_creative_sketch(
            output_folder=self.root,
            remove_sourcefile=False,
            expected_md5=self.md5_sum,
            gdrive_id="1Dstn3Cv2yAnxt1JXC0zPa1VPVDSz9gc-",
            filename="creative_sketch.zip",
        )

    def _load_items_metadata(self):
        """Load and cache metadata for all items in the dataset.

        Creates a parquet file containing metadata for all sketches including
        category, split, global ID, and whether the sample is good.
        If metadata cache exists, loads from the cached file.
        """
        if not os.path.exists(os.path.join(self.root, ".metadata.parquet")):
            # 读取原始数据文件
            bird_json_path = os.path.join(self.root, "creative_sketches", "raw_data_clean/creative_birds_json.txt")
            creature_json_path = os.path.join(self.root, "creative_sketches", "raw_data_clean/creative_creatures_json.txt")
            
            bird_json_base = json.loads(open(bird_json_path).read())
            creature_json_base = json.loads(open(creature_json_path).read())
            
            # 加载所有数据的元数据，不进行筛选
            items_metadata = pd.DataFrame(columns=["category", "split", "id", "is_good"])
            cnt = 0
            
            # 处理鸟类数据
            for sketch in tqdm(bird_json_base, desc="Loading bird metadata"):
                new_row = {
                    "category": "bird",
                    "split": "bird",
                    "id": cnt,
                    "is_good": sketch["good_sample"]
                }
                items_metadata = pd.concat(
                    [items_metadata, pd.DataFrame([new_row])], ignore_index=True
                )
                cnt += 1
            
            # 处理生物数据
            for sketch in tqdm(creature_json_base, desc="Loading creature metadata"):
                new_row = {
                    "category": "creature",
                    "split": "creature",
                    "id": cnt,
                    "is_good": sketch["good_sample"]
                }
                items_metadata = pd.concat(
                    [items_metadata, pd.DataFrame([new_row])], ignore_index=True
                )
                cnt += 1
            
            self.items_metadata = items_metadata
            self.items_metadata.to_parquet(
                os.path.join(self.root, ".metadata.parquet"), compression="zstd"
            )
        else:
            items_metadata = pd.read_parquet(
                os.path.join(self.root, ".metadata.parquet")
            )
            if not set(items_metadata.columns.tolist()) == set(self.metadata):
                os.remove(os.path.join(self.root, ".metadata.parquet"))
                self._load_items_metadata()
            self.items_metadata = items_metadata
        
        # 初始化raw_data列表
        self.raw_data = [None] * len(self.items_metadata)
        # 加载完整的JSON数据
        self._load_json_data()

    def _load_json_data(self):
        # TODO: 合并为gid和subid
        """Load the complete JSON data from the dataset files.
        
        This method reads the bird and creature JSON files and stores
        the data in self.bird_json_base and self.creature_json_base.
        """
        bird_json_path = os.path.join(self.root, "creative_sketches", "raw_data_clean/creative_birds_json.txt")
        creature_json_path = os.path.join(self.root, "creative_sketches", "raw_data_clean/creative_creatures_json.txt")
        
        self.bird_json_base = json.loads(open(bird_json_path).read())
        self.creature_json_base = json.loads(open(creature_json_path).read())
        
        # 创建一个映射，将metadata中的id映射到原始JSON数据中的索引
        self.id_to_sketch_map = {}
        bird_idx = 0
        creature_idx = 0
        
        for idx, row in self.items_metadata.iterrows():
            if row["category"] == "bird":
                self.id_to_sketch_map[idx] = ("bird", bird_idx)
                bird_idx += 1
            else:  # category == "creature"
                self.id_to_sketch_map[idx] = ("creature", creature_idx)
                creature_idx += 1

    def _load_all(self):
        """Load all sketch data into memory if load_all is enabled.

        Loads all sketch data from the JSON files into memory for faster access.
        Only loads if self.load_all is True.
        """
        for idx in tqdm(range(len(self.items_metadata)), desc="Loading all Creative Sketch data to memory"):
            if self.raw_data[idx] is None:
                self.raw_data[idx] = self._load_sketch_data(idx)

    def _load_sketch_data(self, idx):
        """Load a single sketch data by index.
        
        Args:
            idx (int): Index of the sketch to load.
        
        Returns:
            list: List of paths for the sketch.
        """
        # 获取原始JSON数据
        category, orig_idx = self.id_to_sketch_map[idx]
        if category == "bird":
            single_sketch_json = self.bird_json_base[orig_idx]
            colors = COLORS_BIRD
        else:
            single_sketch_json = self.creature_json_base[orig_idx]
            colors = COLORS_CREATURE
        
        # 处理路径数据
        path_list = []
        for j, step in enumerate(single_sketch_json['all_strokes']):
            stroke_label = single_sketch_json["partsUsed"][j]
            stroke_color = colors[stroke_label]

            strokes = []
            for stroke in step:
                if len(stroke) == 0:  # skip the empty stroke
                    continue
                for seg in stroke:
                    p_start = [seg[0], seg[1]]
                    p_end = [seg[2], seg[3]]
                    line = np.stack([p_start, p_end], axis=0)
                    cubic = line_to_cubic(line)
                    cubic._set_vertices_attribute("color", stroke_color)
                    strokes.append(cubic)
            path_list.append(Path(curves=strokes))
        
        return path_list

    def __getitem__(self, idx: int) -> Sketch:
        """Get a sketch by index.

        If a sketch not in memory, load it from disk.

        Args:
            idx (int): Index of the sketch to retrieve.

        Returns:
            Sketch: A Sketch object containing the drawing data as paths.

        Raises:
            IndexError: If index is out of range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # 检查是否已加载到内存，如果没有则加载
        if self.raw_data[idx] is None:
            self.raw_data[idx] = self._load_sketch_data(idx)

        path_list = self.raw_data[idx]
        sketch = Sketch(paths=path_list)
        return sketch
  
    def extra_repr(self) -> str:
        """Return extra information for the string representation.
        
        Returns:
            str: Additional information to include in __repr__.
        """
        num_birds = len(self.items_metadata[self.items_metadata["category"] == "bird"])
        num_creatures = len(self.items_metadata[self.items_metadata["category"] == "creature"])
        num_good = len(self.items_metadata[self.items_metadata["is_good"] == True])
        
        return (
            f"Total samples: {len(self)}\n" \
            f"Bird samples: {num_birds}\n" \
            f"Creature samples: {num_creatures}\n" \
            f"Good samples: {num_good}" \
            f"Bad samples: {len(self) - num_good}"
        )


if __name__ == "__main__":
    # Default Load
    dataset = CreativeSketch()
    
    print(f"Dataset summary:\n{dataset.extra_repr()}")

    # 示例：通过items_metadata选择数据
    # 1. 选择鸟类数据
    bird_indices = dataset.items_metadata[dataset.items_metadata["category"] == "bird"].index
    print(f"Number of bird sketches: {len(bird_indices)}")
    
    # 2. 选择生物数据
    creature_indices = dataset.items_metadata[dataset.items_metadata["category"] == "creature"].index
    print(f"Number of creature sketches: {len(creature_indices)}")
    
    # 3. 选择好的样本
    good_indices = dataset.items_metadata[dataset.items_metadata["is_good"] == True].index
    print(f"Number of good sketches: {len(good_indices)}")
    
    # 4. 组合筛选：鸟类的好样本
    good_bird_indices = dataset.items_metadata[
        (dataset.items_metadata["category"] == "bird") & 
        (dataset.items_metadata["is_good"] == True)
    ].index
    print(f"Number of good bird sketches: {len(good_bird_indices)}")
    
    # 加载一个鸟类的好样本
    if len(good_bird_indices) > 0:
        bird_sketch = dataset[good_bird_indices[0]]
        print(f"Loaded bird sketch with {len(bird_sketch.paths)} paths")
    
    # 加载一个生物的样本
    if len(creature_indices) > 0:
        creature_sketch = dataset[creature_indices[0]]
        print(f"Loaded creature sketch with {len(creature_sketch.paths)} paths")
    
    # 渲染示例
    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer
    
    renderer = CairoRenderer(800, (1, 1, 1))
    if 'bird_sketch' in locals():
        raster_image = renderer.render(bird_sketch)
        outpath = "test.png"
        if outpath is not None:
            raster_image_png = Image.fromarray(raster_image, "RGB")
            raster_image_png.save(outpath, "PNG")
            print(f"Sketch saved to {outpath}")

