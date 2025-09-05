from sketchkit.datasets import Sketchy

if __name__ == '__main__':
    my_data = Sketchy()
    print(len(my_data))
    # print(my_data[0])
    index = 12
    print(f"The file name of the {index}th sketch is: {my_data.sketch_names[index]}")

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer
    renderer = CairoRenderer(512, (1,1,1))
    raster_image = renderer.render(my_data[index])
    outpath = f'test_{index}.png'
    if outpath is not None:
        raster_image_png = Image.fromarray(raster_image, 'RGB')
        raster_image_png.save(outpath, 'PNG')