Sketch Datasets
===============

How to implement a dataset?
---------------------------

The base class is :class:`sketchkit.utils.dataset.SketchDataset`.  
You should implement the following methods in a subclass:

- ``_check_integrity``
- ``_download``
- ``_load_items_metadata``
- ``_load_all``
- ``__getitem__``

By default, the dataset **does not load full data into memory**, only metadata.  
The ``_load_items_metadata`` method should load all metadata into memory for searching later.  
Metadata is recommended to be stored in a ``pandas.DataFrame`` for better indexing.  
You can cache the metadata in a file to speed up subsequent loads.  

``_check_integrity`` is used to validate that the downloaded directories/files are intact.

Dataset Usages
--------------

After the dataset conversion, you can load a dataset with a single line of code:

.. code-block:: python

   # Default load (metadata only)
   dataset = QuickDraw()

   # Load all data into memory at once
   dataset = QuickDraw(load_all=True)

   # Download from CISLAB CDN
   dataset = QuickDraw(root=tmpdir, cislab_source=True)

   # Search items using metadata
   cats = dataset.items_metadata[
       (dataset.items_metadata["category"] == "cat") &
       (dataset.items_metadata["split"] == "train")
   ]

   # Load sketches based on metadata
   cats_sketch = [dataset[row.id] for _, row in cats[:100].iterrows()]

.. currentmodule:: sketchkit.datasets

The following are the built-in sketch datasets currently available in SketchKit:

Available Datasets
--------------

.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    hzySketch
    QuickDraw
    TracingVsFreehand
    ControlSketch


Utility Scripts
--------------

.. autosummary::
    :toctree: generated/
    :template: module.rst

    sketchkit.datasets.fscoco
    sketchkit.datasets.differsketching