Sketch Datasets
===============

Overview
---------------------------
SketchKit provides a variety of **sketch-based datasets** for research on sketch recognition, retrieval, and generation, including most of current popular datasets.   

All datasets share a unified interface, so you can load them, search sketches by metadata, or convert sketches into the standard Sketch format easily.

Metadata
---------------------------
Each dataset object provides an attribute ``items_metadata``, :class:`pandas.DataFrame`, 
which contains useful information for searching and filtering sketches.

``items_metadata`` is your main tool to search, filter, and batch load sketches efficiently.

    - ``id``: unique identifier of each sketch
    - ``category``: sketch category (e.g., "cat", "dog", "chair")
    - ``split``: dataset split (e.g., "train", "test", "val"), available in some datasets

Key concepts of dataset parameters
---------------------------
**root**: ``Datatype: str``, ``Default: "~/.cache/sketchkit/datasets"`` 

    | Directory where the dataset is stored or downloaded. 
    | If the dataset is not present locally, it will be automatically downloaded to this path. 
    | `root` dictionary can be costumed, if not, it will store datasets in the default path. 

    Example: dataset = QuickDraw(root=tmpdir)

**load_all**: ``Datatype: bool``, ``Default: False``

    Whether to load all sketches in the dataset into memory at once.  

    - ``False``: Load metadata only, sketch can be loaded on demand via metadata. 
    - ``True``: Load the entire dataset into memory, having faster access but higher RAM usage. 

    Example: dataset = QuickDraw(load_all=True)

**cislab_source**: ``Datatype: bool``, ``Default: False``

    Whether to download the dataset from the CIS LAB mirror site or original download link. 

    - ``False``: Use default source (e.g., Github/official download link) to download dataset. 
    - ``True``: Apply CIS LAB mirror site to download dataset. 

    Example: dataset = QuickDraw(cislab_source=True)

**download**: ``Datatype: bool``, ``Default: True``

    Automatically download the dataset if it is missing locally.

    - ``True``: As description.
    - ``False``: **Not recommended.** You need to provide the dataset manually, otherwise an error is raised. 

**split**: ``Datatype: str``, ``Default: "train"``

    Select which split of the dataset to load. (e.g., "train", "test", "val")

    Example: cats = dataset.items_metadata(dataset.items_metadata["split"] == "train")

**category**: ``Datatype: list[str]``, ``Default: None``

    Load only specific categories of sketches.  (e.g., “cat”, “dog”, “chair”)

    Example: cats = dataset.items_metadata(dataset.items_metadata["category"] == "cat") 

.. tip::

    - ``load_all`` should be chosen based on dataset size and memory availability: use ``False`` for large datasets.
    - ``split`` and ``categories`` can be combined to load subset you need. 
    - ``cislab_source`` is applicable to most mainstream dataset.  

Loading a dataset
--------------

.. code-block:: python

   # Default load (metadata only)
   dataset = QuickDraw()

   # Load all data into memory at once
   dataset = QuickDraw(load_all=True)

   # Download from CISLAB CDN
   dataset = QuickDraw(root=tmpdir, cislab_source=True)


Searching sketches by metadata
--------------

.. code-block:: python

   # Search items using metadata
   cats = dataset.items_metadata[
       (dataset.items_metadata["category"] == "cat") &
       (dataset.items_metadata["split"] == "train")
   ]

   # Load sketches based on metadata
   cats_sketch = [dataset[row.id] for _, row in cats[:100].iterrows()]

.. currentmodule:: sketchkit.datasets


Available Datasets
--------------

The following are the built-in sketch datasets currently available in SketchKit:

.. autosummary::
    :toctree: generated/
    :template: class_dataset.rst

    hzySketch
    QuickDraw
    ControlSketch
    TUBerlin
    TracingVsFreehand
    OpenSketch
    SketchXPRIS
    Sketchy
    PhotoSketching
    GMUSketchCleanup


Utility Scripts
--------------

.. autosummary::
    :toctree: generated/
    :template: module.rst

    sketchkit.datasets.fscoco
    sketchkit.datasets.differsketching