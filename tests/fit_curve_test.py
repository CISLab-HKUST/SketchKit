from sketchkit.datasets.differsketching import DifferSketching
from sketchkit.utils.transform import convert_to_curve

if __name__ == '__main__':
    my_data = DifferSketching()
    print(len(my_data))
    # print(my_data[0])

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer
    renderer = CairoRenderer(500, (1,1,1))
    raster_image = renderer.render(my_data[1], fit_canvas=True)
    sketch = my_data[1]
    outpath = 'PhotoSketch.png'
    svg_outpath = 'PhotoSketch.svg'
    if outpath is not None:
        sketch.to_svg(svg_outpath)
        raster_image_png = Image.fromarray(raster_image, 'RGB')
        raster_image_png.save(outpath, 'PNG')
    
    new_sketch = convert_to_curve(sketch, datasetName=my_data.get_dataset_name(), maxError=4.0)
    new_sketch.to_svg('PhotoSketch_curve.svg')
    raster_image_curve = renderer.render(new_sketch,fit_canvas=True)
    raster_image_curve_png = Image.fromarray(raster_image_curve, 'RGB')
    raster_image_curve_png.save('PhotoSketch_curve.png', 'PNG')