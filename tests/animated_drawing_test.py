

from sketchkit.datasets import QuickDraw
from sketchkit.ordering import Orderer
from tqdm import tqdm
import pandas as pd

from sketchkit.utils.file import save_seq_gif
if __name__ == '__main__':
    my_data = QuickDraw(cislab_source=True)
    print(len(my_data))
    # print(my_data[0])
    pd.set_option('display.max_colwidth', None)

    print(my_data.items_metadata.loc[10])

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer
    renderer = CairoRenderer(800, (1,1,1))
    raster_image = renderer.render(my_data[10], fit_canvas=False)
    sketch = my_data[10]
    outpath = 'PhotoSketch.png'
    svg_outpath = 'PhotoSketch.svg'
    
    
    if outpath is not None:
        sketch.to_svg(filename=svg_outpath)
        raster_image_png = Image.fromarray(raster_image, 'RGB')
        raster_image_png.save(outpath, 'PNG')

    
    animated_ordering = Orderer.create(method="Fu", device="cpu")
    sketch = animated_ordering.run(sketch)

    sketch_frames = sketch.get_progressive_frames(sample_rate=10)
    raster_images = [renderer.render(frame, fit_canvas=False) for frame in tqdm(sketch_frames)]
    save_seq_gif(raster_images, 'animated_drawing.gif')
