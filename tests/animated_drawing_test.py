

from sketchkit.datasets.differsketching import DifferSketching
from sketchkit.ordering import Orderer
from tqdm import tqdm
if __name__ == '__main__':
    my_data = DifferSketching(cislab_source=True)
    print(len(my_data))
    # print(my_data[0])

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer, render_frames
    renderer = CairoRenderer(800, (1,1,1))
    raster_image = renderer.render(my_data[0], fit_canvas=False)
    sketch = my_data[0]
    outpath = 'PhotoSketch.png'
    svg_outpath = 'PhotoSketch.svg'
    
    animated_ordering = Orderer.create(method="Fu", device="cpu")
    sketch = animated_ordering.run(sketch, 'animated_sketch.gif')
    
    if outpath is not None:
        sketch.to_svg(svg_outpath)
        raster_image_png = Image.fromarray(raster_image, 'RGB')
        raster_image_png.save(outpath, 'PNG')