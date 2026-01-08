# HED image2sketch backend

This directory contains the lightweight assets and wrappers needed to run the
Holistically-Nested Edge Detection (HED) model as an ``image2sketch`` method.
The original HED project by Saining Xie et al. ships a bespoke Caffe fork. For
SketchKit we rely on OpenCV's DNN Caffe loader so inference only requires the
pretrained weights and the exported deployment graph.

## Files

- ``assets/deploy.prototxt`` – the network definition mirrored from the upstream
  ``examples/hed`` directory.
- ``model.py`` – the ``HEDModel`` wrapper implementing the
  ``Image2SketchModel`` protocol.
- ``__init__.py`` – exposes ``HEDModel`` for import convenience.
- ``LICENSE`` – the upstream BSD 2-Clause license.

The 56 MB pretrained weights are downloaded automatically on first use to
``~/.sketchkit/image2sketch/hed/hed_pretrained_bsds.caffemodel``. If you
prefer manual control, place the official ``hed_pretrained_bsds.caffemodel`` in
``sketchkit/image2sketch/hed/assets/`` and the wrapper will copy it to the cache.
You can also pass ``auto_download=False`` with a custom ``checkpoint_dir`` and
prepare the directory yourself before calling ``generate``.

## Usage

```python
from sketchkit.image2sketch import image2sketch

edge_map = image2sketch(image, method="hed", size=512)
```

The ``size`` argument follows the standard ``image2sketch`` contract: pass an
integer for a square output or a ``(width, height)`` tuple for explicit control.
The result is a ``PIL.Image`` instance in ``"L"`` mode containing the edge map.

Upstream project: https://github.com/s9xie/hed
