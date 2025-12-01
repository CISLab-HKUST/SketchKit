import os
import cv2
import torch
from PIL import Image
import gdown
from torch.nn import functional as F
import warnings

warnings.filterwarnings("ignore")

from sketchkit.animation.RIFE.model.RIFE_HDv3 import Model
from sketchkit.utils.file import download_with_wget, CISLAB_CDN, extract_files


class RIFE:
    """
    A class for sketch animation using the method of paper "Real-Time Intermediate Flow Estimation for Video Frame Interpolation"
    in ECCV 2022.

    Attributes:
        device (str): The device that is used.
    """
    def __init__(self, device: str="cuda"):
        """
        Initializes the RIFE animator.

        Args:
            device (str, optional): The device that is used.
        """
        self.method = "RIFE"
        self.device = device

        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        self._download_models()
        self.model = Model(self.model_dir, device)

    def _download_models(self, cislab_source=False):
        """Download the pretrained models"""
        root = os.path.join(
            os.environ.get("HOME", "/"),
            ".cache/sketchkit/models/animation/",
            self.method,
        )
        os.makedirs(root, exist_ok=True)
        self.model_dir = os.path.join(root, "train_log")
        self.model_path = os.path.join(self.model_dir, "flownet.pkl")

        filename = "RIFEv4.25_0919.zip"
        archive_path = os.path.join(root, filename)

        if not os.path.exists(self.model_path):
            if cislab_source:
                cislab_url = f"{CISLAB_CDN}/datasets/Sketchy/{filename}"
                print(f"Attempting download from CISLAB: {cislab_url}")
                download_with_wget(
                    cislab_url, archive_path, desc="Downloading from CISLAB CDN"
                )
                print("CISLAB download completed successfully!")
            else:
                gdrive_id = "1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg"
                print("Using gdown for Google Drive download...")
                gdown.download(id=gdrive_id, output=archive_path, quiet=False)
                print("Google Drive download completed successfully!")

                extract_files(
                    file_path=archive_path, output_dir=root, remove_sourcefile=True
                )
                os.remove(os.path.join(self.model_dir, "IFNet_HDv3.py"))
                os.remove(os.path.join(self.model_dir, "refine.py"))
                os.remove(os.path.join(self.model_dir, "RIFE_HDv3.py"))

    def run(self, input_image_list: list[str], inner_frames: int):
        """
        Generate sketch animation or inbetweening for a list of keyframes.

        Args:
            input_image_list (list[str]): a list of paths to keyframes.
            inner_frames (int): number of intermediate frames.

        Returns:
            frame_list (numpy.ndimage): a list of image frames in size H*W*3 in [0, 255].
        """
        frame_list = []
        for img_i in range(len(input_image_list) - 1):
            img0_path = input_image_list[img_i]
            img1_path = input_image_list[img_i + 1]

            img0 = cv2.imread(img0_path, cv2.IMREAD_UNCHANGED)
            img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
            img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(self.device) / 255.0).unsqueeze(0)
            img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(self.device) / 255.0).unsqueeze(0)

            n, c, h, w = img0.shape
            ph = ((h - 1) // 64 + 1) * 64
            pw = ((w - 1) // 64 + 1) * 64
            padding = (0, pw - w, 0, ph - h)
            img0 = F.pad(img0, padding)
            img1 = F.pad(img1, padding)

            if img_i == 0:
                frame_list.append(img0)
            n = inner_frames + 1
            for i in range(n - 1):
                frame_list.append(self.model.inference(img0, img1, (i + 1) * 1.0 / n))
            frame_list.append(img1)

            frame_list = [(item[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
                          for item in frame_list]
        return frame_list
