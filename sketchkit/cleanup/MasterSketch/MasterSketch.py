import os
import torch
from git.util import join_path
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image

from sketchkit.cleanup.MasterSketch.model.model_gan import model, immean, imstd
from sketchkit.utils.file import download_with_wget, CISLAB_CDN


class MasterSketch:
    """
    A class for rough sketch cleanup using the method of paper "Mastering Sketching: Adversarial Augmentation for Structured Prediction"
    in SIGGRAPH 2018.

    Attributes:
        device (str): The device that is used.
    """
    def __init__(self, device: str="cuda"):
        """
        Initializes the MasterSketch cleaner.

        Args:
            device (str, optional): The device that is used.
        """
        self.method = "MasterSketch"
        self.device = device

        self._download_models()
        self.model = model
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def _download_models(self):
        """Download the pretrained models
        """
        root = os.path.join(
            os.environ.get("HOME", "/"),
            ".sketchkit/weights/cleanup/",
            self.method,
        )
        filename = "model_gan.pth"
        self.model_path = os.path.join(root, filename)
        if not os.path.exists(self.model_path):
            cislab_url = f"{CISLAB_CDN}/weights/MasterSketch/{filename}"
            print(f"Attempting download from CISLAB: {cislab_url}")
            download_with_wget(
                cislab_url, self.model_path, desc="Downloading from CISLAB CDN"
            )
            print("CISLAB download completed successfully!")

    def run(self, image_path: str, out_folder: str = "cleanup"):
        """
        Sketch simplification for the input raster image.

        Args:
            image_path (str): a path to an input raster image.
            out_folder (str, optional): an output folder to store the output file.

        Returns:
            pred_out (numpy.ndimage): a sketch image in H*W in [0, 255].
        """
        img_base, img_name = os.path.split(image_path)

        data = Image.open(image_path).convert('L')
        w, h = data.size[0], data.size[1]
        pw = 8 - (w % 8) if w % 8 != 0 else 0
        ph = 8 - (h % 8) if h % 8 != 0 else 0

        data = ((transforms.ToTensor()(data) - immean) / imstd).unsqueeze(0)
        if pw != 0 or ph != 0:
            data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data

        pred = model.forward(data)[0]  # (1, H, W)
        pred_out = (pred.data.cpu().numpy()[0] * 255.0).astype(np.uint8)  # (H, W)

        if out_folder is not None:
            out_dir = os.path.join(img_base, out_folder)
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, img_name)
            cv2.imwrite(save_path, pred_out)

        return pred_out
