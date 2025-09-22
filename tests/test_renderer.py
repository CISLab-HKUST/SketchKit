from sketchkit.datasets import OpenSketch #, TUBerlin

if __name__ == '__main__':
    my_data = OpenSketch()
    print(len(my_data))

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer
    renderer = CairoRenderer(2000, (1,1,1))
    raster_image = renderer.render_with_texture(sketch=my_data[0],texture='./sketchkit/assets/grid.png',width=3)
    temp = renderer.render(sketch=my_data[0])
    outpath = 'SketchXPRIS_grid.png'
    if outpath is not None:
        raster_image_png = Image.fromarray(temp, 'RGB')
        raster_image.save(outpath, 'PNG')
        raster_image_png.save('SketchXPRIS_no_texture.png', 'PNG')