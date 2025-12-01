import numpy as np
from PIL import Image

from sketchkit.vectorization import Vectorizer


if __name__ == "__main__":
    vectorizer = Vectorizer(method="DeepVecSIG24")
    # vectorizer = Vectorizer(method="LineDrawer")

    img_path = "26_0.png"
    img_np = np.array(Image.open(img_path).convert("L"))
    sketch = vectorizer.run(img_np)

    # indir = "test"
    # vectorizer.run(indir)

    # from PIL import Image
    # from sketchkit.renderer.cairo_renderer import CairoRenderer
    #
    # renderer = CairoRenderer(1024, (1, 1, 1))
    # raster_image = renderer.render(sketch=sketch)
    # outpath = 'test_vectorize.png'
    # raster_image_png = Image.fromarray(raster_image, 'RGB')
    # raster_image_png.save(outpath, 'PNG')
