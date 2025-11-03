import os
import cv2
import torch
from PIL import Image
import gdown
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

from sketchkit.animation.model.RIFE_HDv3 import Model
from sketchkit.utils.file import download_with_wget, CISLAB_CDN, extract_files


class Animator:
    """
    A class for sketch animation or inbetweening.

    Attributes:
        method (str): The vectorization method that is used.
        device (str): The device that is used.
    """
    def __init__(self, method: str ="RIFE", device: str ="cuda"):
        """
        Initializes the Animator with the name of the method.

        Args:
            method (str, optional): The vectorization method that is used.
            device (str, optional): The device that is used.
        """
        self.method = method
        self.device = device

        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        # TODO: can add other vectorization methods
        if method == "RIFE":
            self._download_models()
            self.model = Model(self.model_dir, device)
        else:
            raise Exception('TBD')

    def _download_models(self, cislab_source=False):
        """Download the pretrained models
        """
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

    def _gen_gif(self, all_files, save_path):
        """
        Gennerate .gif for images in a folder.
        """
        all_files += all_files[::-1][1:]

        gif_frames = []
        for img_path in all_files:
            img_i = Image.open(img_path)
            gif_frames.append(img_i)

        first_frame = gif_frames[0]
        first_frame.save(save_path, save_all=True, append_images=gif_frames, loop=0, duration=0.01)

    def run(self, input_0: str, input_1: str, inner_frames: int = 8, out_folder: str = "animation"):
        """
        Generate sketch animation or inbetweening.

        Args:
            input_0 (str): a path to an input raster image (frame 0).
            input_1 (str): a path to an input raster image (frame 1).
            inner_frames (int): number of intermediate frames.
            out_folder (str, optional): an output folder to store the output svg file.
        """
        image_list = [input_0, input_1]

        img_list = []
        output_base = os.path.join('./', out_folder)
        os.makedirs(output_base, exist_ok=True)

        for img_i in range(len(image_list) - 1):
            img0_path = image_list[img_i]
            img1_path = image_list[img_i + 1]

            img0 = cv2.imread(img0_path, cv2.IMREAD_UNCHANGED)
            img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
            img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)
            img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)

            n, c, h, w = img0.shape
            ph = ((h - 1) // 64 + 1) * 64
            pw = ((w - 1) // 64 + 1) * 64
            padding = (0, pw - w, 0, ph - h)
            img0 = F.pad(img0, padding)
            img1 = F.pad(img1, padding)

            if img_i == 0:
                img_list.append(img0)
            n = inner_frames + 1
            for i in range(n - 1):
                img_list.append(self.model.inference(img0, img1, (i + 1) * 1. / n))
            img_list.append(img1)

        _, filename = os.path.split(input_0)
        filename = filename.split('.')[0]

        filepaths = []
        for i in range(len(img_list)):
            save_path = os.path.join(output_base, filename + '_img{}.png'.format(i))
            cv2.imwrite(save_path,
                        (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
            filepaths.append(save_path)

        save_gif_path = os.path.join(output_base, filename + '_dynamic.gif')
        self._gen_gif(filepaths, save_gif_path)
