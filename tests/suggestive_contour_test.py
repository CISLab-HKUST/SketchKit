from sketchkit.renderer.cairo_renderer import render_suggestive_countour
import numpy as np

render_suggestive_countour('media/rapid.obj', 'test1.ppm', np.array([0, -1, 0, 70, 0, 0, 1, -70, -1, 0, 0, -461, 0, 0, 0, 1]))

