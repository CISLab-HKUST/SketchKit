import glob
import ssl
import urllib.request
import xml.etree.ElementTree as ET

import wget
from svg.path import parse_path, path

from sketchkit.core import Sketch
from sketchkit.core.sketch import Path, Curve, Vertex, Point
from sketchkit.utils.general_utils.extract_files import *


def download_gmu_sketch_cleanup(output_folder, remove_sourcefile=True):
    """Download the GMU-Sketch-Cleanup dataset.

    Args:
        output_folder: The directory to save the downloaded dataset.
        remove_sourcefile: Whether to remove the source zip file after extraction.
    """
    print("The download of GMU-Sketch-Cleanup has started......")
    os.makedirs(output_folder, exist_ok=True)

    url = "https://cragl.cs.gmu.edu/sketchbench/Benchmark_Dataset.zip"
    zip_path = os.path.join(output_folder, "gmu_sketch_cleanup.zip")

    # Attempt normal download with SSL verification.
    try:
        wget.download(url, out=output_folder)
        print("Normal SSL download succeeded.")
    except Exception as e:
        msg = str(e)
        print("Normal download failed:", msg)

        # If SSL verification fails, retry with SSL verification disabled.
        if "SSL" in msg or "CERTIFICATE_VERIFY_FAILED" in msg:
            print("Retrying with SSL verification disabled (insecure) + progress bar...")

            # Create a temporary opener with SSL verification disabled.
            ctx = ssl._create_unverified_context()
            opener = urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ctx)
            )
            urllib.request.install_opener(opener)

            # Progress bar for download.
            def _progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = downloaded / total_size * 100
                    print(
                        f"\rDownloading (insecure)... {percent:5.1f}%  "
                        f"({downloaded / 1e6:.1f}/{total_size / 1e6:.1f} MB)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\rDownloading (insecure)... {downloaded / 1e6:.1f} MB",
                        end="",
                        flush=True,
                    )

            try:
                urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
                print("Insecure download succeeded.")
            except Exception as e2:
                print("Insecure download also failed:", e2)
                return
        else:
            return

    print("GMU-Sketch-Cleanup is downloaded!")

    # Extract the downloaded zip file.
    try:
        extract_files(
            file_name=zip_path,
            output_folder=os.path.join(output_folder, "gmu_sketch_cleanup"),
            remove_sourcefile=remove_sourcefile,
        )
    except Exception as e:
        print("Extract failed:", e)
        return

    print("GMU-Sketch-Cleanup is extracted!")


INVALID_SVG_SHAPES = {"rect", "circle", "ellipse", "line", "polyline", "polygon"}


def parse_gmu_svg(svg_file):
    """Parse an SVG file into width, height, path list, and total segment number.

    Args:
        svg_file: The path to the SVG file.

    Returns:
        A tuple containing:
        - (width, height): The dimensions of the SVG canvas.
        - path_list: A list of paths, where each path contains strokes.
        - total_segment_num: The total number of segments in the SVG file.

    Raises:
        Exception: If unsupported elements like transforms or arcs are encountered.
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()

    # Parse dimensions.
    view_box = root.get("viewBox")
    if view_box:
        view_x, view_y, view_w, view_h = [float(x) for x in view_box.split()]
        width = int(view_w - view_x)
        height = int(view_h - view_y)
    else:
        width = int(float(root.get("width", "0").replace("px", "")) or 0)
        height = int(float(root.get("height", "0").replace("px", "")) or 0)

    path_list = []
    total_segment_num = 0

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag in INVALID_SVG_SHAPES:
            continue
        if tag == "path":
            if "transform" in elem.attrib:
                raise Exception(f"Transform not supported in {svg_file}")
            d = elem.attrib.get("d")
            if not d:
                continue
            ps = parse_path(d)
            cps = []
            for seg in ps:
                if isinstance(seg, path.Move):
                    if len(cps) > 1:
                        # Start a new path.
                        seg_num = (len(cps) - 1) // 3
                        strokes = [cps[i * 3:i * 3 + 4] for i in range(seg_num)]
                        path_list.append(strokes)
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
            if len(cps) > 1:
                seg_num = (len(cps) - 1) // 3
                strokes = [cps[i * 3:i * 3 + 4] for i in range(seg_num)]
                path_list.append(strokes)
                total_segment_num += seg_num

    return (width, height), path_list, total_segment_num


class GMUSketchCleanup:
    """A dataset reader for the GMU Rough Sketch Cleanup dataset.

    The dataset directory structure is expected to be:
    Benchmark_Dataset/
      ├── GT/
      ├── Rough/JPG/
      ├── Rough/PNG/
      └── Rough/SVG/
      └── sketch_tags.csv

    Args:
        data_root: The root directory for storing the dataset.
        split: The dataset split, either "Rough" or "GT".
        modality: The data modality, currently only "SVG" is supported.

    Attributes:
        items: A list of file paths for the selected split and modality.
    """

    def __init__(self,
                 data_root=".cache/sketchkit/datasets",
                 split="Rough",
                 modality="SVG"):
        self.data_root = data_root
        self.split = split
        self.modality = modality

        bench_root = os.path.join(data_root, "gmu_sketch_cleanup/Benchmark_Dataset")

        if not os.path.isdir(bench_root):
            download_gmu_sketch_cleanup(data_root)

        if self.split == "GT":
            self.items = sorted(glob.glob(os.path.join(bench_root, "GT", "**", "*.svg"), recursive=True))
        elif self.split == "Rough":
            if self.modality == "SVG":
                self.items = sorted(glob.glob(os.path.join(bench_root, "Rough", "SVG", "**", "*.svg"), recursive=True))
            else:
                raise ValueError("modality must be SVG for now")
        else:
            raise ValueError("split must be one of {Rough,GT}")

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.items)

    def __getitem__(self, idx) -> Sketch:
        """Get a Sketch object for the given index.

        Args:
            idx: The index of the item.

        Returns:
            A Sketch object constructed from the SVG file.

        Raises:
            TypeError: If the current item is not an SVG file.
        """
        svg_path = self.items[idx]
        if not svg_path.lower().endswith(".svg"):
            raise TypeError("Current item is not SVG; cannot construct Sketch")

        (w, h), path_list, _ = parse_gmu_svg(svg_path)
        paths = []
        for p in path_list:
            strokes = []
            for seg in p:
                p0 = Vertex(seg[0][0], seg[0][1])
                p3 = Vertex(seg[3][0], seg[3][1])
                c1 = Point(seg[1][0], seg[1][1])
                c2 = Point(seg[2][0], seg[2][1])
                strokes.append(Stroke(p0, p3, c1, c2))
            paths.append(Path(strokes=strokes))
        return Sketch(paths=paths)
