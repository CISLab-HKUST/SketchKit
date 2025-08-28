from sketchkit.datasets import TUBerlin #, QuickDraw

if __name__ == '__main__':
    my_data = TUBerlin()
    print(len(my_data))

    from PIL import Image
    from sketchkit.renderer.naive_rasterization import Renderer
    renderer = Renderer(512, (1,1,1))
    raster_image = renderer.render(my_data[0])
    outpath = 'test.png'
    if outpath is not None:
        raster_image_png = Image.fromarray(raster_image, 'RGB')
        raster_image_png.save(outpath, 'PNG')