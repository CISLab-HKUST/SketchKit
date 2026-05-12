Supported Datasets
===============

Overview
---------------------------
SketchKit provides a variety of **sketch-based datasets** for research on sketch recognition, retrieval, and generation, including most of current popular datasets.   

All datasets share a unified interface, so you can load them, search sketches by metadata, or convert sketches into the standard Sketch format easily.

.. _dataset-metadata:

Metadata
---------------------------
Each dataset object provides an attribute ``items_metadata``, :class:`pandas.DataFrame`, 
which contains useful information for searching and filtering sketches.

``items_metadata`` is your main tool to search, filter, and batch load sketches efficiently.

    - ``id``: unique identifier of each sketch.
    - ``sub_id``: optional sub-identifier used when one original item contains multiple sketches or derived samples.
    - ``category``: sketch category (e.g., "cat", "dog", "chair").
    - ``split``: dataset split (e.g., "train", "test", "val"), available in some datasets.
    - ...

.. _key-concepts-of-dataset-parameters:

Key concepts of dataset parameters
---------------------------
**root**: ``Datatype: str``, ``Default: "~/.sketchkit/datasets"`` 

    | Directory where the dataset is stored or downloaded. 
    | If the dataset is not present locally, it will be automatically downloaded to this path. 
    | `root` dictionary can be costumed, if not, it will store datasets in the default path. 

    Examples:

    .. code-block:: python

        # Store the dataset in a temporary directory
        dataset = ControlSketch(root=tmpdir)

        # Store the dataset in a custom local directory
        dataset = ControlSketch(root="./controlsketch")

**load_all**: ``Datatype: bool``, ``Default: False``

    Whether to load all sketches in the dataset into memory at once.  

    - ``False``: Load metadata only, sketch can be loaded on demand via metadata. 
    - ``True``: Load the entire dataset into memory, having faster access but higher RAM usage. 

    Example: 
    
    .. code-block:: python

        dataset = ControlSketch(load_all=True)

**download**: ``Datatype: bool``, ``Default: True``

    Automatically download the dataset if it is missing locally.

    - ``True``: As description.
    - ``False``: **Not recommended.** You need to provide the dataset manually, otherwise an error is raised. 

**split**: ``Datatype: str``, ``Default: "train"``

    Select which split of the dataset to load. (e.g., "train", "test", "val")

    Example: 
    
    .. code-block:: python
      
        validation = dataset.items_metadata[dataset.items_metadata["split"] == "validation"]

**category**: ``Datatype: list[str]``, ``Default: None``

    Load only specific categories of sketches.  (e.g., “cat”, “dog”, “chair”)

    Example: 
    
    .. code-block:: python
      
        bears = dataset.items_metadata[dataset.items_metadata["category"] == "bear"]

.. tip::

    - ``load_all`` should be chosen based on dataset size and memory availability: use ``False`` for large datasets.
    - ``split`` and ``categories`` can be combined to load subset you need. 

Loading a dataset
--------------

.. code-block:: python

   # Default load (metadata only)
   dataset = ControlSketch()

   # Load all data into memory at once
   dataset = ControlSketch(load_all=True)


Searching sketches by metadata
--------------

.. code-block:: python

   # Search items using metadata
   bears = dataset.items_metadata[
            (dataset.items_metadata["category"] == "bear")
            & (dataset.items_metadata["split"] == "validation")
        ]

   # Load sketches based on metadata
   bears_sketch = [dataset[row.id] for _, row in bears[:100].iterrows()]

.. currentmodule:: sketchkit.datasets

.. _available-datasets:

Available Datasets
--------------

The following are the built-in sketch datasets currently available in SketchKit. Each dataset has its
own manual page in this section so you can customize descriptions and examples.

.. toctree::
  :maxdepth: 1

  datasets/LineDrawer
  datasets/QuickDraw
  datasets/ControlSketch
  datasets/TUBerlin
  datasets/TracingVsFreehand
  datasets/OpenSketch
  datasets/SketchXPRIS
  datasets/Sketchy
  datasets/PhotoSketching
  datasets/GMUSketchCleanup
  datasets/FSCOCO
  datasets/DifferSketching
  datasets/CreativeSketch
  datasets/SketchIME

.. _dataset-feature:

Dataset Feature
------------------------------

The following table shows features of each built-in SketchKit dataset.

- Vector representation (polyline or Bézier curve) used in the original datasets.

.. list-table::
    :widths: 1 1
    :header-rows: 1
    :class: fullwidth

    * - Dataset
      - Representation
    * - :class:`LineDrawer <sketchkit.datasets.LineDrawer>`
      - Polyline
    * - :class:`QuickDraw <sketchkit.datasets.QuickDraw>`
      - Polyline
    * - :class:`ControlSketch <sketchkit.datasets.ControlSketch>`
      - Bézier
    * - :class:`TUBerlin <sketchkit.datasets.TUBerlin>`
      - Bézier
    * - :class:`TracingVsFreehand <sketchkit.datasets.TracingVsFreehand>`
      - Polyline
    * - :class:`OpenSketch <sketchkit.datasets.OpenSketch>`
      - Polyline
    * - :class:`SketchXPRIS <sketchkit.datasets.SketchXPRIS>`
      - Polyline
    * - :class:`Sketchy <sketchkit.datasets.Sketchy>`
      - Bézier
    * - :class:`PhotoSketching <sketchkit.datasets.PhotoSketching>`
      - Bézier
    * - :class:`GMUSketchCleanup <sketchkit.datasets.GMUSketchCleanup>`
      - Bézier
    * - :class:`FSCOCO <sketchkit.datasets.FSCOCO>`
      - Polyline
    * - :class:`DifferSketching <sketchkit.datasets.DifferSketching>`
      - Polyline
    * - :class:`CreativeSketch <sketchkit.datasets.CreativeSketch>`
      - Polyline
    * - :class:`SketchIME <sketchkit.datasets.SketchIME>`
      - Polyline

How to contribute a new dataset to SketchKit
---------------------------
You can integrate a new or customized dataset into SketchKit following the :doc:`step-by-step document <3_dataset_contribution>`.