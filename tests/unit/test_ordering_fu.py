import pytest
import importlib

from sketchkit.core.sketch import Sketch, Path, Curve, Point
from sketchkit.ordering.Fu import AnimatedDrawer


def make_sketch(n):
    return Sketch(width=128, height=128, attributes={"author": "test"}, paths=[
        Path([Curve(Point(10, y), Point(118, y), Point(40, y), Point(88, y))])
        for y in range(10, 10 + 20 * n, 20)
    ])


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_small_sketch_preserves_every_stroke(n):
    sketch = make_sketch(n)
    ordered = AnimatedDrawer().run(sketch)
    assert len(ordered.paths) == n
    assert {id(p) for p in ordered.paths} == {id(p) for p in sketch.paths}
    assert ordered.attributes == sketch.attributes


def test_equal_cost_branches_can_share_the_queue(monkeypatch):
    fu = importlib.import_module("sketchkit.ordering.Fu")
    monkeypatch.setattr(fu, "cost_transition", lambda *args: (0, 0))
    ordered = AnimatedDrawer().run(make_sketch(3))
    assert ordered.path_num == 3


def test_empty_path_does_not_shift_stroke_indices():
    sketch = make_sketch(3)
    empty = Path([])
    sketch.paths.insert(0, empty)
    ordered = AnimatedDrawer().run(sketch)
    assert {id(p) for p in ordered.paths} == {id(p) for p in sketch.paths}
    assert ordered.paths[-1] is empty


def test_infeasible_graph_reports_actionable_error():
    with pytest.raises(ValueError, match="max_k"):
        AnimatedDrawer(max_k=0).run(make_sketch(3))


def test_directed_graph_can_start_after_stroke_zero(monkeypatch):
    fu = importlib.import_module("sketchkit.ordering.Fu")
    # Three horizontal strokes must be traversed bottom to top: 2 -> 1 -> 0.
    monkeypatch.setattr(fu, "compute_T_junctions", lambda a, b: int(a[2] < b[2]))
    original = make_sketch(3)
    ordered = AnimatedDrawer().run(original)
    assert ordered.paths == list(reversed(original.paths))
