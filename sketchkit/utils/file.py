import os

import numpy as np
import cv2
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

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    for img in img_seq:
        if isinstance(img, Image.Image):
            frame = np.asarray(img.convert("RGB"))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame = np.asarray(img)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))

        out.write(frame)

    out.release()
    print("Saved video to", file_path)


def save_seq_gif(img_seq, file_path, fps=10, w=None, h=None, loop=0):
    """
    Save a sequence of images as an animated GIF using PIL.
    """
    if not hasattr(img_seq, "__getitem__"):
        img_seq = list(img_seq)

    if len(img_seq) == 0:
        raise ValueError("img_seq is empty")

    frames = []
    for img in img_seq:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))

        if w is not None and h is not None:
            if img.size != (w, h):
                img = img.resize((w, h), Image.Resampling.LANCZOS)

        frames.append(img)

    if not frames:
        return

    duration = 1000 / fps
    frames[0].save(
        file_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )
    print("Saved video to", file_path)
