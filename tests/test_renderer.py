from sketchkit.datasets.tracing_vs_freehand import TracingVsFreehand

if __name__ == '__main__':
    my_data = TracingVsFreehand()
    print(len(my_data))

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer
    renderer = CairoRenderer(2000, (1,1,1))
    # raster_image = renderer.render_with_texture(sketch=my_data[0],texture='./sketchkit/assets/grid.png',width=3)
    temp = renderer.render(sketch=my_data[0])
    outpath = 'SketchXPRIS_grid.png'
    if outpath is not None:
        temp = Image.fromarray(temp, 'RGB')
        temp.save(outpath, 'PNG')
        temp.save('tvf.png', 'PNG')
