import os
import cv2
import torch
from PIL import Image
import gdown
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")

from sketchkit.animation.model.RIFE_HDv3 import Model
from sketchkit.utils.file import extract_files


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

    def _download_models(self):
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

        if not os.path.exists(self.model_path):
            gdrive_id = "1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg"
            output_path = os.path.join(root, "RIFEv4.25_0919.zip")
            gdown.download(id=gdrive_id, output=output_path, quiet=False)

            extract_files(
                file_path=output_path, output_dir=root, remove_sourcefile=True
            )
            os.remove(os.path.join(self.model_dir, "IFNet_HDv3.py"))
            os.remove(os.path.join(self.model_dir, "refine.py"))
            os.remove(os.path.join(self.model_dir, "RIFE_HDv3.py"))

    def _gen_gif(self, img_dir):
        all_files = os.listdir(img_dir)
        all_files.sort()
        all_files = [item for item in all_files if '.png' in item]

        frame_num = len(all_files)
        all_files = ['img' + str(frame_i) + '.png' for frame_i in range(frame_num)]
        all_files += all_files[::-1][1:]

        gif_frames = []
        for img_name in all_files:
            img_i = Image.open(os.path.join(img_dir, img_name))
            gif_frames.append(img_i)

        save_path = os.path.join(img_dir, 'dynamic.gif')
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

        for i in range(len(img_list)):
            cv2.imwrite(os.path.join(output_base, 'img{}.png'.format(i)),
                        (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

        self._gen_gif(output_base)
