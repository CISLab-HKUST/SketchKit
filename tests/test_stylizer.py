from sketchkit.datasets import PhotoSketching
from sketchkit.stylization import Stylizer

dataset = PhotoSketching()
sketch = dataset[0]

stylizer = Stylizer(
    method="NeuralBrushstroke",
    model_name="style2",
    style_id="playdoh10",
)

# 1) Stylize from a Sketch
stylized_image = stylizer.run(
    sketch,
    output_path="stylized_output.png",
    canvas_size=512,
)

# 2) Stylize from a PNG geometry image
png_result = stylizer.run_from_png(
    "input_sketch.png",
    output_path="stylized_from_png.png",
    style_id="misc05",
)

# 3) Blend two styles
blend_result = stylizer.run(
    sketch,
    output_path="stylized_blend.png",
    style_id="playdoh10",
    style_id2="misc05",
    style_blend_alpha=0.3,
)