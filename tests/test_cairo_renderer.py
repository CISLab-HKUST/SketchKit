from sketchkit.datasets import DifferSketching
import random

if __name__ == "__main__":
    my_data = DifferSketching(skip_integrity_check=True)
    print(len(my_data))

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer

    renderer = CairoRenderer()
    # raster_image = renderer.render_with_texture(sketch=my_data[0],texture='./sketchkit/assets/grid.png',width=3)
    raster_image = renderer.render(my_data[0], {"random_color": True})
    raster_image.save("test1111111.png")
    # for i in range(10):
    #     raster_image = renderer.render(
    #         sketch=my_data[i],
    #         render_options={
    #             "canvas_size": (
    #                 1024 + random.randint(0, 100),
    #                 1024 + random.randint(0, 100),
    #             ),
    #             "stroke_width": random.randint(1, 10),
    #             "background_color": (1.0, 1.0, 1.0),
    #             "stroke_color": (random.random(), 0.0, 0.0),
    #         },
    #     )
    #     outpath = f"DifferSketching_{i}.png"
    #     # raster_image_png = Image.fromarray(raster_image, "RGB")
    #     raster_image.save(outpath, "PNG")
    # temp = renderer.render(
    #     sketch=my_data[33],
    #     render_options={
    #         "canvas_size": (768, 768),
    #         "stroke_width": 10,
    #         "background_color": (0.5, 0.4, 0.3),
    #         "stroke_color": (0.6, 0.7, 0.8),
    #     },
    # )
    # outpath = "SketchXPRIS_grid.png"
    # temp.save(outpath, "PNG")
    # temp.save("tvf.png", "PNG")
