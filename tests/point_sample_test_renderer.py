from sketchkit.datasets import TUBerlin
import cv2
import numpy as np

if __name__ == '__main__':
    my_data = TUBerlin(cislab_source=True)

    from PIL import Image
    from sketchkit.renderer.cairo_renderer import CairoRenderer
    renderer = CairoRenderer(512, (1,1,1))
    raster_image = renderer.render(my_data[1], 10)
    outpath = 'PhotoSketch.png'
    if outpath is not None:
        raster_image_png = Image.fromarray(raster_image, 'RGB')
        raster_image_png.save(outpath, 'PNG')
        sketch = np.zeros((raster_image.shape[0], raster_image.shape[1], 4), dtype=np.uint8)
        for path in my_data[1].paths:
            ts = np.linspace(0, 1, 1000)
            points = path.points(ts)  # (N, 2)
            # print(points[:5])
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                cv2.line(sketch, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 255, 255, 255), 10)
        # is sketch are all one?
        print(np.all(sketch[:, :, 3] == 255))
        cv2.imwrite('sketch.png', sketch)