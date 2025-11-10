

from sketchkit.datasets.differsketching import DifferSketching
from sketchkit.models import animated_drawing, render_animated_drawing
from tqdm import tqdm
if __name__ == '__main__':
    my_data = DifferSketching()
    print(len(my_data))
    # print(my_data[0])

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer
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
    for i, frame in enumerate(tqdm(sketch_frames)):
        raster_image = renderer.render(frame, fit_canvas=False)
        raster_image_png = Image.fromarray(raster_image, 'RGB')
        raster_image_png.save(f'tests/frames/frame_{i}.png', 'PNG')

    # 合并所有帧为一个 GIF
    frames = [Image.open(f'tests/frames/frame_{i}.png') for i in range(len(sketch_frames))]
    frames[0].save('animated_sketch.gif', save_all=True, append_images=frames[1:], duration=100, loop=0)