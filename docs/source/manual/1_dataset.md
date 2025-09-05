# Sketch Datasets

## How to implement a dataset?

```{eval-rst}
The base class is :class:`sketchkit.utils.dataset.SketchDataset`. You should implement `_check_integrity`, `_download`, `_load_items_metadata`, `_load_all`, `__getitem__` methods.

By default, the dataset doesn't load data to memory, only metadata.`_load_items_metadata` should load all metadata of items into memory, which can be used for searching later. Metadata should be stored in a `pandas.DataFrame` for better indexing. You can caching the metadata using a file to load after the first time.

`_check_integrity` is to validate the downloaded dir/files is intact.

An example implementation is `sketchkit.datasets.QuickDraw`:

.. autoclass:: sketchkit.datasets.QuickDraw
   :members:
   :private-members:
   :special-members: __getitem__
```

## Dataset Usages

After the dataset conversion, we can load a dataset with a single-line code:

```python
   # Default Load
   dataset = QuickDraw()

   # Load all data to memory at once
   dataset = QuickDraw(load_all=True)

   # Download from CISLAB CDN
   dataset = QuickDraw(root=tmpdir, cislab_source=True)

   # Search items using metadata
   cats = dataset.items_metadata[
      (dataset.items_metadata["category"] == "cat")
      & (dataset.items_metadata["split"] == "train")
   ]

   # Load sketches
   cats_sketch = [dataset[row.id] for _, row in cats[:100].iterrows()]
```