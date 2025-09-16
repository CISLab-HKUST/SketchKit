from sketchkit.datasets import TUBerlin

if __name__ == '__main__':
    my_data = TUBerlin()
    print(len(my_data))
    # print(my_data[0])

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer
    renderer = CairoRenderer(500, (1,1,1))
    raster_image = renderer.render(my_data[1])
    sketch = my_data[1]
    outpath = 'PhotoSketch.png'
    svg_outpath = 'PhotoSketch.svg'
    if outpath is not None:
        sketch.to_svg(svg_outpath)
        raster_image_png = Image.fromarray(raster_image, 'RGB')
        raster_image_png.save(outpath, 'PNG')
    for path in sketch.paths:
        print(path.curve_lengths())
    curve = sketch.paths[0].curves[0]
    print(curve.is_line())
    print(curve.points)
