from sketchkit.datasets import OpenSketch #, TUBerlin
import gdown

if __name__ == '__main__':
    my_data = OpenSketch()
    print(len(my_data))
    print(my_data[0])

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer
    renderer = CairoRenderer(512, (1,1,1))
    raster_image = renderer.render(my_data[0])
    outpath = 'test.png'
    if outpath is not None:
        raster_image_png = Image.fromarray(raster_image, 'RGB')
        raster_image_png.save(outpath, 'PNG')