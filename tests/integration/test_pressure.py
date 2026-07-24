import pytest

from sketchkit.datasets import TracingVsFreehand
from sketchkit.renderer import CairoRenderer


pytestmark = pytest.mark.integration


def test_pressure_dataset_sample_can_render():
    try:
        dataset = TracingVsFreehand()
    except Exception as exc:
        pytest.skip(f"TracingVsFreehand not available locally: {exc}")

    p29_rows = dataset.items_metadata[
        dataset.items_metadata["participant_id"] == "p29"
    ]
    if len(p29_rows) < 2:
        pytest.skip("not enough participant p29 samples")

    sketch = dataset[int(p29_rows.iloc[1].id)]
    renderer = CairoRenderer()
    image = renderer.render(sketch, {"stroke_color": (0, 0, 0)})

    assert sketch.path_num > 0
    assert image.size[0] > 0 and image.size[1] > 0
