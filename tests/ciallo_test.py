
from sketchkit.datasets import DifferSketching #, TUBerlin
from PIL import Image
from sketchkit.renderer.ciallo_renderer import CialloRenderer


if __name__ == "__main__":
    my_data = DifferSketching()
    print(len(my_data))
    
    renderer = CialloRenderer(1024)    
    renderer.render(my_data[13013], "textured", thickness=7.0, spacing=1.0, color=(0,1,1,1),
                    rotation_randomness=0.5, noise_factor=0.2,
                    brush_texture_path="./sketchkit/assets/stamp2.png", output_path="ciallo_stamp2_1.png")