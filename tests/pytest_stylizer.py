import os
import pytest
from sketchkit.datasets import PhotoSketching
from sketchkit.stylization import Stylizer


def test_stylizer_run_and_png_conversion(tmp_path):
    try:
        dataset = PhotoSketching()
    except Exception as exc:
        pytest.skip(f"PhotoSketching dataset unavailable: {exc}")

    assert len(dataset) > 0
    sketch = dataset[0]

    plane_png = os.path.join("test_images", "plane.png")
    if not os.path.exists(plane_png):
        pytest.skip("Stylizer PNG input test_images/plane.png is missing")

    stylizer = Stylizer(
        method="NeuralBrushstroke",
        model_name="style2",
        style_id="playdoh10",
    )

    sketch_output = tmp_path / "sketch_output.png"
    stylized_image = stylizer.run(
        sketch,
        output_path=str(sketch_output),
        canvas_size=512,
    )
    assert stylized_image is not None
    assert sketch_output.exists()

    png_output = tmp_path / "image_output.png"
    png_result = stylizer.run_from_png(
        plane_png,
        output_path=str(png_output),
        style_id="misc05",
    )
    assert png_result is not None
    assert png_output.exists()

    blend_output_path = tmp_path / "blend_output.png"
    blend_result = stylizer.run(
        sketch,
        output_path=str(blend_output_path),
        style_id="playdoh10",
        style_id2="misc05",
        style_blend_alpha=0.3,
    )
    assert blend_result is not None
    assert blend_output_path.exists()
