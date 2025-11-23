
from sketchkit.datasets import OpenSketch #, TUBerlin

if __name__ == '__main__':
    my_data = OpenSketch()
    print(len(my_data))

    from PIL import Image
    from sketchkit.renderer.ciallo_renderer import CialloRenderer
    renderer = CialloRenderer(2000, (1,1,1))
    raster_image = renderer.render(
        sketch=my_data[1],
        texture_path="./sketchkit/assets/stamp2.png",  # your brush/footprint image (RGBA recommended)
        canvas_size=1024,
        stroke_width_px=12.0,        # short dimension equals this width
        spacing_px=6.0,             # center-to-center spacing in pixels
        out_path="output.png",
        clear_color=(0, 0, 0, 0),    # transparent
    )
