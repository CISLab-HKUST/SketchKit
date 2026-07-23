from sketchkit.datasets import PhotoSketching
from sketchkit.stylization import Stylizer

if __name__ == "__main__":
    dataset = PhotoSketching(cislab_source=True)
    sketch = dataset[0]

    stylizer = Stylizer(
        method="NeuralBrushstroke",
        model_name="style2",
        style_id="playdoh10",
    )
    output_path = "outputs/stylizer"
    # 1) Stylize from a Sketch
    stylized_image = stylizer.run(
        sketch,
        output_path=f"{output_path}/sketch_output.png",
        canvas_size=512,
    )

    # 2) Stylize from a PNG geometry image
    png_result = stylizer.run_from_png(
        "test_files/plane.png",
        output_path=f"{output_path}/image_output.png",
        style_id="misc05",
    )

    # 3) Blend two styles
    blend_result = stylizer.run(
        sketch,
        output_path=f"{output_path}/blend_output.png",
        style_id="playdoh10",
        style_id2="misc05",
        style_blend_alpha=0.3,
    )
