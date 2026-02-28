

from sketchkit.datasets import CreativeSketch
from tqdm import tqdm
import pandas as pd

from sketchkit.utils.file import save_seq_gif
if __name__ == '__main__':
    my_data = CreativeSketch(cislab_source=True)
    print(len(my_data))
    # print(my_data[0])
    pd.set_option('display.max_colwidth', None)

    print(my_data.items_metadata.loc[10])

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions
    
    renderer = CairoRenderer(CairoRenderOptions(
        background_color=(1.0, 1.0, 1.0)
    ))
    
    sketch = my_data[10]
    outpath = 'PhotoSketch.png'
    svg_outpath = 'PhotoSketch.svg'
    
    raster_image = renderer.render(my_data[10])
    if outpath is not None:
        sketch.to_svg(filename=svg_outpath)
        
        raster_image.save(outpath, 'PNG')
