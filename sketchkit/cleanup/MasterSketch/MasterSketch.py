import os
import torch
from git.util import join_path
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import gdown

from sketchkit.cleanup.MasterSketch.model.model_gan import model, immean, imstd

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
        """Download the pretrained models from Google Drive."""
        root = os.path.join(
            os.environ.get("HOME", "/"),
            ".sketchkit/weights/cleanup/",
            self.method,
        )
        os.makedirs(root, exist_ok=True)

        filename = "model_gan.pth"
        self.model_path = os.path.join(root, filename)

        if not os.path.exists(self.model_path):
            gdrive_id = "1QdDr1AhyqFaX6ZWWxPs2YJVkulanKjYK"
            print("Downloading MasterSketch model from Google Drive...")
            gdown.download(id=gdrive_id, output=self.model_path, quiet=False)
            print("Google Drive download completed successfully!")

    def run(
        self,
        image_array: np.ndarray,
        image_folder: str = "./",
        out_folder: str = "cleanup"
    ) -> np.ndarray:
        """
        Sketch simplification for the input raster image.

        Args:
            image_array (numpy.ndarray): input raster image in shape (H, W), with values in [0, 255].
            image_folder (str, optional): the folder of the input raster image.
            out_folder (str, optional): an output folder to store the output file.

        Returns:
            pred_out (numpy.ndimage): a sketch image in H*W in [0, 255].
        """
        w, h = image_array.shape[1], image_array.shape[0]
        pw = 8 - (w % 8) if w % 8 != 0 else 0
        ph = 8 - (h % 8) if h % 8 != 0 else 0

        data = ((transforms.ToTensor()(image_array) - immean) / imstd).unsqueeze(0)
        if pw != 0 or ph != 0:
            data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data

        pred = model.forward(data)[0]  # (1, H, W)
        pred_out = (pred.data.cpu().numpy()[0] * 255.0).astype(np.uint8)  # (H, W)

        if out_folder is not None:
            out_dir = os.path.join(image_folder, out_folder)
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, "result.png")
            cv2.imwrite(save_path, pred_out)

        return pred_out
