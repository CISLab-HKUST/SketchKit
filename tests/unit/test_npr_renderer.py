import math
import os
from sketchkit.renderer.npr_renderer import NPRRenderer, NPRRenderOptions
from sketchkit.utils.file import save_seq_gif
from sketchkit.core.camera import Camera

if __name__ == "__main__":
    output_path = "outputs/npr_renderer"
    os.makedirs(output_path, exist_ok=True)
    renderer = NPRRenderer(obj_path="test_files/head.obj")
    cameras = []
    r = 2.5
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        camera = Camera()
        camera.set_look_at([0, 0, 0])
        camera.set_xyz([r * math.cos(rad), r * math.sin(rad), 0])
        cameras.append(camera)

    print(f"Starting render for {len(cameras)} cameras...")

    print("Rendering with suggestive contour method...")
    options_1 = NPRRenderOptions(method="SuggestiveContour", threshold=0.005)
    images = renderer._render(cameras, options_1)
    gif_path = f"{output_path}/SuggestiveContour.gif"
    save_seq_gif(images, str(gif_path))

    print("Rendering with suggestive contour method...")
    options_1 = NPRRenderOptions(method="RidgesAndValleys", threshold=0.1)
    images = renderer._render(cameras, options_1)
    gif_path = f"{output_path}/RidgesAndValleys.gif"
    save_seq_gif(images, str(gif_path))

    print("Rendering with suggestive contour method...")
    options_1 = NPRRenderOptions(method="AppearantRidges", threshold=0.1)
    images = renderer._render(cameras, options_1)
    gif_path = f"{output_path}/AppearantRidges.gif"
    save_seq_gif(images, str(gif_path))

    print(f"Render completed, generated {len(images)} images.")
