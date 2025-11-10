

from sketchkit.datasets.differsketching import DifferSketching
from sketchkit.ordering import animated_drawing, render_animated_drawing
from tqdm import tqdm
if __name__ == '__main__':
    my_data = DifferSketching()
    print(len(my_data))
    # print(my_data[0])

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer, render_frames
    renderer = CairoRenderer(800, (1,1,1))
    raster_image = renderer.render(my_data[0], fit_canvas=False)
    sketch = my_data[0]
    outpath = 'PhotoSketch.png'
    svg_outpath = 'PhotoSketch.svg'
    if outpath is not None:
        sketch.to_svg(svg_outpath)
        raster_image_png = Image.fromarray(raster_image, 'RGB')
        raster_image_png.save(outpath, 'PNG')
    animated_path_indices = animated_drawing(sketch)
    print(animated_path_indices)
    sketch_frames = render_animated_drawing(sketch, animated_path_indices)
    render_frames(sketch_frames, 'animated_sketch.gif', canvas_size=800, canvas_color=(1, 1, 1), width=4)
