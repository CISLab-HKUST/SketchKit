import os
import cv2
import torch
from PIL import Image

from sketchkit.animation.RIFE import RIFE
from sketchkit.animation.LineDrawer import LineDrawerAnimator


class Animator:
    """
    A class for sketch animation or inbetweening.

    Attributes:
        method (str): The animation method that is used.
        device (str): The device that is used.
    """

    def __init__(self, method: str = "RIFE", device: str = "cuda"):
        """
        Initializes the Animator with the name of the method.

        Args:
            method (str, optional): The animation method that is used.
            device (str, optional): The device that is used.
        """
        self.method = method
        self.device = device

        if method == "RIFE":
            self.animate_method = RIFE(device)
        elif method == "LineDrawer":
            self.animate_method = LineDrawerAnimator(device=device)
        else:
            raise Exception("Please add implementations.")

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
        first_frame.save(
            save_path, save_all=True, append_images=gif_frames, loop=0, duration=0.01
        )

    def run(
        self,
        input_0: str,
        input_1: str,
        inner_frames: int = 8,
        output_folder: str | None = None,
    ) -> list[torch.Tensor]:
        """
        Generate sketch animation or inbetweening.

        Args:
            input_0 (str): a path to an input raster image (frame 0).
            input_1 (str): a path to an input raster image (frame 1).
            inner_frames (int): number of intermediate frames.
            output_folder (str, optional): an output folder to store the output svg file.
        """
        input_image_list = [input_0, input_1]
        img_list = self.animate_method.run(input_image_list, inner_frames)

        if output_folder is not None:

            _, filename = os.path.split(input_0)
            filename = filename.split(".")[0]

            output_base = os.path.join("./", output_folder)
            os.makedirs(output_base, exist_ok=True)
            filepaths = []
            for i in range(len(img_list)):
                save_path = os.path.join(output_base, filename + "_img{}.png".format(i))
                cv2.imwrite(save_path, img_list[i])
                filepaths.append(save_path)

            save_gif_path = os.path.join(output_base, filename + "_dynamic.gif")
            self._gen_gif(filepaths, save_gif_path)

        return img_list
