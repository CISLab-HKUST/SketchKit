import os
import gdown
import traceback
import xml.etree.ElementTree as ET
from svg.path import parse_path, path
from sketchkit.core import Sketch
from sketchkit.core.sketch import Path, Curve, Vertex, Point
from sketchkit.utils.general_utils.md5sum import *
from sketchkit.utils.general_utils.extract_files import *


def download_photosketch_dataset(
    output_folder, 
    remove_sourcefile: bool = True,
    expected_md5: str | None = None,
    gdrive_id: str = "1_AIxKnZXQms5Ezb-cEeVIDIoVG-eliHc",
    filename: str = "all-in-one.zip",
):
    """Download the complete PhotoSketch dataset as a single all-in-one.zip file.
    
    This downloads the complete dataset archive that contains:
    - get_images.py: Python script for downloading 1,000 outdoor photos
    - imagemeta.csv: Image metadata of 1,000 outdoor photos
    - LICENSE: License file
    - list/: Dataset split files
    - sketch/: SVG format sketches (5 sketches per photo, with stroke timestamps)
    - sketch-rendered/: PNG format rendered sketches (each SVG sketch rendered with 3 different line widths)
    - Readme.txt: Documentation
    
    Args:
        output_folder: The folder to download the dataset.
        remove_sourcefile: Whether to remove the source file after extraction.
        expected_md5: The expected MD5 checksum of the downloaded file.
        gdrive_id: The Google Drive file ID for the all-in-one archive.
        filename: The name of the downloaded file.
    
    Returns:
        None
    """
    print("Downloading PhotoSketch dataset...")
    
    os.makedirs(output_folder, exist_ok=True)
    archive_path = os.path.join(output_folder, filename)
    
    try:
        print("Using gdown for Google Drive download...")
        # Prefer gdown when available, it handles confirm tokens and cookies internally.
        file_id = gdrive_id.split('/')[-2] if '/' in gdrive_id else gdrive_id
        gdown.download(id=file_id, output=archive_path, quiet=False)
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
            file_name=archive_path,
            output_folder=os.path.join(output_folder, "photosketch"),
            remove_sourcefile=remove_sourcefile,
        )
        print("PhotoSketch dataset is extracted!")
    except Exception as e:
        print(e)
        return


# SVG parsing constants and functions
INVALID_SVG_SHAPES = {"rect", "circle", "ellipse", "line", "polyline", "polygon"}


def parse_photosketch_svg(svg_file):
    """Parse an SVG file into width, height, path list, total segment number, and path attributes.
    
    Args:
        svg_file: The path to the SVG file.
        
    Returns:
        A tuple containing:
        - (width, height): The dimensions of the SVG canvas.
        - path_list: A list of paths, where each path contains strokes.
        - total_segment_num: The total number of segments in the SVG file [Segment here is curve in sketchkit].
        - path_attributes: A list of dictionaries containing path attributes.
        
    Raises:
        Exception: If unsupported elements like transforms or arcs are encountered.
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()
    
    # Parse dimensions
    view_box = root.get("viewBox")
    if view_box:
        view_x, view_y, view_w, view_h = [float(x) for x in view_box.split()]
        width = int(view_w - view_x)
        height = int(view_h - view_y)
    else:
        width = int(float(root.get("width", "0").replace("px", "")) or 0)
        height = int(float(root.get("height", "0").replace("px", "")) or 0)
    
    path_list = []
    path_attributes = []  # path in sketchkit has attributes like color and thickness
    total_segment_num = 0
    
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag in INVALID_SVG_SHAPES:
            continue
        if tag == "path":
            if "transform" in elem.attrib:
                raise Exception(f"Transform not supported in {svg_file}")
            
            # extract path attributes (only keep color and thickness)
            # convert color value to RGB array format
            stroke_color = elem.get('stroke', '#000000')
            if stroke_color == '#000' or stroke_color == '#000000':
                rgb_color = (0.0, 0.0, 0.0)  # black
            elif stroke_color == '#fff' or stroke_color == '#ffffff':
                rgb_color = (1.0, 1.0, 1.0)  # white
            elif stroke_color.startswith('#'):
                # handle other hexadecimal colors
                hex_color = stroke_color.lstrip('#')
                if len(hex_color) == 3:
                    hex_color = ''.join([c+c for c in hex_color])
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                rgb_color = (r, g, b)
            else:
                rgb_color = (0.0, 0.0, 0.0)  # default black
            
            # original geometry parsing code
            d = elem.attrib.get("d")
            if not d:
                continue
            ps = parse_path(d)
            cps = []
            
            # extract attributes of current path (path in svg file)
            attrs = {
                'stroke': rgb_color,    # color of path in svg file
                'stroke-width': float(elem.get('stroke-width', 1)), # thickness of <path> in svg file
                't': elem.get('t', ''),  # timestamp information of <path> in svg file, not used in sketchkit
                'id': elem.get('id', '')  # id of <path> in svg file
            }
            
            for seg in ps:
                if isinstance(seg, path.Move):
                    if len(cps) > 1:
                        # start a new path
                        seg_num = (len(cps) - 1) // 3
                        strokes = [cps[i * 3:i * 3 + 4] for i in range(seg_num)] # stroke here is curve in sketchkit, so stroke[s] here is path in sketchkit
                        path_list.append(strokes)
                        path_attributes.append(attrs)  # add corresponding attributes, the length of path_attributes is the same as path_list
                        total_segment_num += seg_num
                    cps = [(seg.start.real, seg.start.imag)]
                elif isinstance(seg, path.CubicBezier):
                    s, c1, c2, e = seg.start, seg.control1, seg.control2, seg.end
                    cps.append((c1.real, c1.imag))
                    cps.append((c2.real, c2.imag))
                    cps.append((e.real, e.imag))
                elif isinstance(seg, path.Line):
                    s, e = seg.start, seg.end
                    sx, sy, ex, ey = s.real, s.imag, e.real, e.imag
                    c1x = 2 / 3 * sx + 1 / 3 * ex
                    c1y = 2 / 3 * sy + 1 / 3 * ey
                    c2x = 1 / 3 * sx + 2 / 3 * ex
                    c2y = 1 / 3 * sy + 2 / 3 * ey
                    cps.append((c1x, c1y))
                    cps.append((c2x, c2y))
                    cps.append((ex, ey))
                elif isinstance(seg, path.Arc):
                    raise Exception("Arc not supported")
                else:
                    raise Exception("Unknown segment type", type(seg))
            
            # Handle the last path if there are remaining control points
            if len(cps) > 1:
                seg_num = (len(cps) - 1) // 3
                strokes = [cps[i * 3:i * 3 + 4] for i in range(seg_num)]
                path_list.append(strokes)
                path_attributes.append(attrs)  # add corresponding attributes, the length of path_attributes is the same as path_list
                total_segment_num += seg_num
    
    return (width, height), path_list, total_segment_num, path_attributes


class PhotoSketch:
    """The PhotoSketch dataset loader.
    
    PhotoSketch is a dataset for photo-to-sketch generation, containing paired images
    and their corresponding sketch representations. The dataset includes:
    - 1,000 outdoor photos
    - 5,000 SVG format sketches (5 sketches per photo, with stroke timestamps)
    - 15,000 PNG format rendered sketches (each SVG sketch rendered with 3 different line widths)
    
    Args:
        data_base: The root directory for storing the dataset.
        file_id: The Google Drive file ID for the all-in-one archive.
        archive_name: The name of the downloaded file.
        md5: The expected MD5 checksum of the downloaded file.
        remove_sourcefile: Whether to remove the source file after extraction.
    
    Example:
        >>> dataset = PhotoSketch()
        >>> print(f"Dataset contains {len(dataset)} items")
        >>> sketch = dataset[0]  # Get first sketch as Sketch object
        >>> print(f"Sketch has {sketch.path_num} paths")
        >>> print(f"Sketch has {sketch.curve_num} curves")
    """
    
    def __init__(
        self,
        data_base: str = ".cache/sketchkit/datasets",
        file_id: str = "1_AIxKnZXQms5Ezb-cEeVIDIoVG-eliHc",
        archive_name: str = "all-in-one.zip",
        md5: str | None = None,
        remove_sourcefile: bool = True,
    ):
        """Initialize the PhotoSketch dataset."""
        data_root = os.path.join(os.environ.get("HOME"), data_base)
        os.makedirs(data_root, exist_ok=True)
        self.photosketch_root = os.path.join(data_root, "photosketch")
        
        if not os.path.isdir(self.photosketch_root):
            download_photosketch_dataset(
                output_folder=data_root,
                remove_sourcefile=remove_sourcefile,
                expected_md5=md5,
                gdrive_id=file_id,
                filename=archive_name,
            )
        else:
            print("PhotoSketch Database already exists.")
        
        # Verify dataset structure
        if not self._check_sketch_dataset_exists():
            raise FileNotFoundError(
                f"PhotoSketch sketch dataset not found at {self.photosketch_root}. "
                "Please download the dataset manually."
            )
        
        # Build sketch dataset index
        self._build_sketch_index()

        # TODO: Initialize paths for other formats (for future use)
        self.sketch_png_dir = os.path.join(self.photosketch_root, "sketch-rendered")
        self.photo_dir = os.path.join(self.photosketch_root, "image")
        
        # TODO: Initialize empty lists for other formats (for future use)
        self.photo_files = []
        self.sketch_png_files = []

        print("PhotoSketch sketch dataset loaded successfully!")
    
    def _check_sketch_dataset_exists(self) -> bool:
        """Check if the dataset exists with the expected structure."""
        # Only check for sketch directory since that's what we need for SketchKit
        self.sketch_svg_dir = os.path.join(self.photosketch_root, "sketch")
        
        if not os.path.isdir(self.sketch_svg_dir):
            print(f"Required directory not found: {self.sketch_svg_dir}")
            return False
        
        # Check if there are actual SVG files in the sketch directory
        svg_files = []
        for root, dirs, files in os.walk(self.sketch_svg_dir):
            svg_files.extend([f for f in files if f.lower().endswith('.svg')])
        
        if len(svg_files) == 0:
            print(f"Warning: No SVG files found in {self.sketch_svg_dir}")
            return False
        
        print(f"Found {len(svg_files)} SVG files in sketch dataset")
        return True
    
    def _build_sketch_index(self):
        """Build the dataset index."""
        # Focus only on SVG sketches since that's what we need for SketchKit
        
        if os.path.isdir(self.sketch_svg_dir):
            # Find all SVG files
            self.sketch_svg_files = []
            for root, dirs, files in os.walk(self.sketch_svg_dir):
                for file in files:
                    if file.lower().endswith('.svg'):
                        rel_path = os.path.relpath(os.path.join(root, file), self.sketch_svg_dir)
                        self.sketch_svg_files.append(rel_path)
            
            self.sketch_svg_files.sort()
            print(f"Indexed {len(self.sketch_svg_files)} SVG files")
        else:
            self.sketch_svg_files = []
            print("Warning: No sketch directory found")
        
    
    def __len__(self) -> int:
        """Return the number of SVG sketches in the dataset."""
        return len(self.sketch_svg_files)
    
    def __getitem__(self, idx: int) -> Sketch:
        """Get a Sketch object for the given index.
        
        Args:
            idx: Index of the sketch to retrieve.
            
        Returns:
            A Sketch object constructed from the SVG file.
            
        Raises:
            IndexError: If the index is out of range.
            Exception: If SVG parsing fails.
        """
        if idx >= len(self.sketch_svg_files):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.sketch_svg_files)} items")
        
        sketch_svg_path = os.path.join(self.sketch_svg_dir, self.sketch_svg_files[idx])
        
        # Verify the SVG file exists
        if not os.path.exists(sketch_svg_path):
            raise FileNotFoundError(f"SVG file not found: {sketch_svg_path}")
        
        # Parse the SVG file and construct a Sketch object
        try:
            (w, h), path_list, _, path_attributes = parse_photosketch_svg(sketch_svg_path)
            
            # Validate that we have valid paths
            if not path_list:
                raise ValueError(f"No valid paths found in SVG file: {sketch_svg_path}")
            
            paths = []
            for i, p in enumerate(path_list):
                strokes = []
                for seg in p:
                    # Validate segment has exactly 4 control points
                    if len(seg) != 4:
                        raise ValueError(f"Invalid segment in SVG: expected 4 points, got {len(seg)}")
                    
                    p0 = Vertex(seg[0][0], seg[0][1])
                    p3 = Vertex(seg[3][0], seg[3][1])
                    c1 = Point(seg[1][0], seg[1][1])
                    c2 = Point(seg[2][0], seg[2][1])
                    strokes.append(Curve(p0, p3, c1, c2))
                
                # create Path
                path = Path(curves=strokes)
                
                # set Path level attributes (will be automatically passed to all Curves and Vertices)
                if i < len(path_attributes):
                    attrs = path_attributes[i]
                    path.color = attrs['stroke']
                    path.thickness = attrs['stroke-width']
                
                paths.append(path)
            
            # create Sketch
            sketch = Sketch(paths=paths)
            
            # 如果所有path都有相同的属性，可以设置Sketch级别（可选）
            # 这里我们让每个path保持自己的属性，所以不设置Sketch级别
            
            return sketch
        except Exception as e:
            raise Exception(f"Failed to parse SVG file {sketch_svg_path}: {str(e)}")
    
    def get_sketch_svg_path(self, idx: int) -> str:
        """Get the path to an SVG sketch file."""
        if idx >= len(self.sketch_svg_files):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.sketch_svg_files)} items")
        
        return os.path.join(self.sketch_svg_dir, self.sketch_svg_files[idx])
    
    def get_sketch_png_path(self, idx: int) -> str:
        """Get the path to a PNG rendered sketch file."""
        if len(self.sketch_png_files) == 0:
            raise ValueError("No PNG files available in this dataset")
        if idx >= len(self.sketch_png_files):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.sketch_png_files)} items")
        
        return os.path.join(self.sketch_png_dir, self.sketch_png_files[idx])
    
    def get_photo_path(self, idx: int) -> str:
        """Get the path to the corresponding original image."""
        # This would need to be implemented based on the actual file naming convention
        # For now, return a placeholder path (file may not exist)
        return os.path.join(self.photo_dir, f"image_{idx:04d}.jpg")
