

from sketchkit.datasets import QuickDraw
from sketchkit.ordering import Orderer
from tqdm import tqdm
import pandas as pd
if __name__ == '__main__':
    my_data = QuickDraw(cislab_source=True)
    pd.set_option('display.max_colwidth', None)
    print(my_data.items_metadata.loc[0])
    # print(my_data.metadata[0])
    # print(len(my_data))
    # print(my_data[0])

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer
    renderer = CairoRenderer(800, (1,1,1))
    raster_image = renderer.render(my_data[0], fit_canvas=False)
    sketch = my_data[0]
    outpath = 'myPhotoSketch.png'
    svg_outpath = 'myPhotoSketch.svg'
    
    
    
    if outpath is not None:
        sketch.to_svg(svg_outpath)
        raster_image_png = Image.fromarray(raster_image, 'RGB')
        raster_image_png.save(outpath, 'PNG')

    animated_ordering = Orderer.create(method="Fu", device="cpu")
    sketch = animated_ordering.run(sketch, 'animated_sketch.gif')