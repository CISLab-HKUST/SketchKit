import os

import numpy as np
import av
from PIL import Image
import wget
import hashlib
from tqdm import tqdm
from pyunpack import Archive
import gdown

CISLAB_CDN = "https://cislab.hkust-gz.edu.cn/projects/sketchkit"


def download_with_wget(
    url: str, file_path: str, overwrite=True, pg_bar=True, desc=None
):
    """Downloads a file from a URL with a progress bar using tqdm.

    Args:
        url: The URL to download the file from.
        file_path: The local path where the file will be saved.
        overwrite: Whether to overwrite the file if it already exists. Defaults to True.
        desc: Description text to show in the progress bar. Defaults to None.
    """
    if overwrite and os.path.exists(file_path):
        os.remove(file_path)

    if pg_bar:
        last = [0]  # previous 'current' from wget
        with tqdm(
            desc=desc,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
            dynamic_ncols=True,
            initial=0,
            total=100,
        ) as bar:

            def hook(current, total, width):
                if total and (bar.total is None or bar.total != total):
                    bar.total = total
                    bar.refresh()  # Refresh to show the full bar properly
                bar.update(current - last[0])
                last[0] = current

            wget.download(url, out=file_path, bar=hook)
    else:
        wget.download(url, out=file_path, bar=None)


def download_with_gdown(
    output_folder,
    gdrive_id: str,
    filename: str,
):
    """Download dataset from google drive.
    Args:
        output_folder: The folder to download the dataset.
        gdrive_id: The Google Drive ID of the dataset.
        filename: The name of the downloaded file.

    Returns:
        None
    """

    os.makedirs(output_folder, exist_ok=True)
    archive_path = os.path.join(output_folder, filename)
    if os.path.exists(archive_path):
        os.remove(archive_path)

    try:
        print("Using gdown for Google Drive download...")
        # Prefer gdown when available, it handles confirm tokens and cookies internally.
        gdown.download(id=gdrive_id, output=archive_path, quiet=False)
        print("\nDownload finished!")
    except Exception as e:
        traceback.print_exc()
        print("Download failed.")
        return


def extract_files(file_path, output_dir, remove_sourcefile=True):
    """Extracts files from an archive to a specified folder.

    Args:
        file_path: Path to the archive file to extract.
        output_dir: Path to the folder where files will be extracted.
        remove_sourcefile: Whether to remove the source archive file after extraction.
            Defaults to True.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        Archive(file_path).extractall(output_dir)
    except Exception as e:
        raise e

    print("{} has been extracted succesfully!".format(file_path))

    if not remove_sourcefile:
        return

    try:
        os.remove(file_path)
    except Exception as e:
        raise e

    print("Source file has been removed succesfully!")


def dir_md5(path, ignore_hidden=True):
    """Calculates the MD5 hash of all files in a directory.

    Args:
        path: Path to the directory.
        ignore_hidden: Whether to ignore hidden files (starting with '.'). Defaults to True.

    Returns:
        str: The MD5 hash as a hexadecimal string.
    """
    md5 = hashlib.md5()
    all_files = []
    for root, _, files in os.walk(path):
        for f in files:
            if ignore_hidden and f.startswith("."):
                continue
            filepath = os.path.join(root, f)
            all_files.append(os.path.relpath(filepath, path))  # normalize

    for filepath in sorted(all_files):  # ensure global ordering
        with open(os.path.join(path, filepath), "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                md5.update(chunk)
    return md5.hexdigest()


def file_md5(file_path):
    """Calculates the MD5 hash of a single file.

    Args:
        file_path: Path to the file.

    Returns:
        str: The MD5 hash as a hexadecimal string.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def save_seq_video(img_seq, file_path, fps, w=None, h=None):
    if not hasattr(img_seq, "__getitem__"):
        img_seq = list(img_seq)

    if len(img_seq) == 0:
        raise ValueError("img_seq is empty")

    if w is None or h is None:
        first = img_seq[0]
        if isinstance(first, Image.Image):
            w, h = first.size
        else:
            arr = np.asarray(first)
            if arr.ndim < 2:
                raise ValueError("Cannot infer dimensions from the provided image")
            h, w = int(arr.shape[0]), int(arr.shape[1])

    output = av.open(file_path, "w")
    stream = output.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"

    for img in img_seq:
        if isinstance(img, Image.Image):
            img = np.asarray(img.convert("RGB"))
        else:
            img = np.asarray(img)

        frame = av.VideoFrame.from_ndarray(img, format="rgb24")

        if frame.width != w or frame.height != h or frame.format.name != stream.pix_fmt:
            frame = frame.reformat(width=w, height=h, format=stream.pix_fmt)

        for packet in stream.encode(frame):
            output.mux(packet)

    for packet in stream.encode():
        output.mux(packet)
    output.close()
    print("Saved video to", file_path)


def save_seq_gif(img_seq, file_path, fps=10, w=None, h=None, loop=0):
    """
    Save a sequence of images as an animated GIF using PyAV.
    
    Fixes segmentation fault by performing palette quantization with PIL 
    before creating the PyAV frame.
    """
    # Make sure we can index
    if not hasattr(img_seq, "__getitem__"):
        img_seq = list(img_seq)

    if len(img_seq) == 0:
        raise ValueError("img_seq is empty")

    # Infer size
    if w is None or h is None:
        first = img_seq[0]
        if isinstance(first, Image.Image):
            w, h = first.size
        else:
            arr = np.asarray(first)
            if arr.ndim < 2:
                raise ValueError("Cannot infer dimensions from the provided image")
            h, w = int(arr.shape[0]), int(arr.shape[1])

    # Open GIF container
    output = av.open(file_path, mode="w", format="gif")

    stream = output.add_stream("gif", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "pal8"

    for img in img_seq:
        # 1. Ensure we have a PIL Image
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        
        # 2. Resize if necessary to match the stream dimensions
        # (Avoids PyAV trying to resize a paletted frame later)
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.LANCZOS)

        # 3. Convert to 'P' mode (Palette) using PIL's adaptive quantization
        # This generates the palette safely in Python land.
        img_p = img.convert("P", palette=Image.ADAPTIVE, colors=256)
        
        # 4. Extract indices and palette for PyAV
        # indices: numpy array of pixel values (0-255)
        img_indices = np.asarray(img_p)
        
        # palette: PyAV expects shape (256, 4) in ARGB or RGBA. 
        # PIL gives a flat list of RGB values [r, g, b, r, g, b, ...]
        pil_palette = img_p.getpalette()
        
        # Pad PIL palette with zeros if it has fewer than 256 colors
        if len(pil_palette) < 768:
            pil_palette.extend([0] * (768 - len(pil_palette)))
            
        # Reshape to (256, 3) -> RGB
        palette_rgb = np.array(pil_palette, dtype=np.uint8).reshape((256, 3))
        
        # Add Alpha channel (255 = Opaque) to make it (256, 4) -> RGBA/ARGB
        # PyAV usually expects the data in the order: A, R, G, B (uint8 bytes)
        # However, for 'pal8', PyAV handles the endianness, but providing 4 bytes is key.
        alpha = np.full((256, 1), 255, dtype=np.uint8)
        palette_rgba = np.hstack((alpha, palette_rgb))

        # 5. Create the frame directly in 'pal8' format
        # We pass a tuple: (pixel_indices, palette)
        frame = av.VideoFrame.from_ndarray((img_indices, palette_rgba), format="pal8")
        
        # 6. Encode
        for packet in stream.encode(frame):
            output.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        output.mux(packet)

    output.close()
    print("Saved video to", file_path)
