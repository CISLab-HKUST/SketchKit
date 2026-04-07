

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
    from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions
    render_options = CairoRenderOptions(
        canvas_size=(800, 800),
        background_color=(1.0, 1.0, 1.0)
    )
    renderer = CairoRenderer(render_options)
    
    sketch = my_data[10]
    outpath = 'PhotoSketch.png'
    svg_outpath = 'PhotoSketch.svg'
    
    
    

    
    animated_ordering = Orderer(method="Fu", device="cpu")
    sketch = animated_ordering.run(sketch)

    sketch_frames = sketch.get_progressive_frames(length_per_frame=10)
    raster_images = [renderer.render(frame) for frame in tqdm(sketch_frames)]
    for i, raster_image in enumerate(raster_images):
        raster_image.save(f'media/frames/frame_{i:04d}.png', 'PNG')
    save_seq_gif(raster_images, 'animated_drawing.gif')
    raster_image = renderer.render(my_data[10])
    if outpath is not None:
        sketch.to_svg(filename=svg_outpath)
        
        raster_image.save(outpath, 'PNG')
