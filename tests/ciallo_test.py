
from sketchkit.datasets import DifferSketching #, TUBerlin
from PIL import Image
from sketchkit.renderer.ciallo_renderer import CialloRenderer

if __name__ == '__main__':
    my_data = DifferSketching()
    print(len(my_data))

    renderer = CialloRenderer(800, (1,1,1,1))
    renderer.render(
        sketch=my_data[0],
        brush_image_path="./sketchkit/assets/stamp2.png",
        rotation_randomness=0.2,
        thickness=8.0,
        noise_factor=0.1,
        interval=0.8,
        color=(0.0, 0.0, 1.0, 1.0),
        output_path="stamp.png",
    )
