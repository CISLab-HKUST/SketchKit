import numpy as np
import pytest
from PIL import Image
from sketchkit.vectorization import Vectorizer


def test_vectorizer_line_drawer(tmp_path, data_dir):
    img_path = data_dir / "ordering" / "input.png"
    if not img_path.exists():
        pytest.skip("Vectorization test image is missing")

    img_np = np.array(Image.open(img_path).convert("L"))
    vectorizer = Vectorizer(method="LineDrawer")
    sketch = vectorizer.run(img_np)

    assert sketch is not None
    out_svg = tmp_path / "pikachu.svg"
    sketch.to_svg(filename=str(out_svg))
    assert out_svg.exists()
