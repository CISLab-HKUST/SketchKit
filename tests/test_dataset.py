import psutil
from rich.console import Console
import time

from sketchkit.datasets import PhotoSketching

console = Console()

console.print("QuickDraw Dataset Test", style="bold blue")
console.print()

# console.print("1. Default Load", style="green")
# dataset = QuickDraw()
# # show brief information of the dataset
# console.print(dataset)
# process = psutil.Process()
# memory_mb = process.memory_info().rss / 1024 / 1024
# console.print(f"Current memory usage: {memory_mb:.2f} MB")

# # search data with "category = cat" and "split = train"
# cats = dataset.items_metadata[
#     (dataset.items_metadata["category"] == "cat")
#     & (dataset.items_metadata["split"] == "train")
# ]
# start_time = time.time()
# cats_sketch = [dataset[row.id] for _, row in cats[:100].iterrows()]
# console.print(f"Loading 100 sketches in {time.time() - start_time}")
# del dataset, cats

# console.print()

# console.print("2. Load All", style="green")
# dataset = QuickDraw(load_all=True)
# # show brief information of the dataset
# console.print(dataset)
# process = psutil.Process()
# memory_mb = process.memory_info().rss / 1024 / 1024
# console.print(f"Current memory usage: {memory_mb:.2f} MB")
# # search data with "category = cat" and "split = test"
# cats = dataset.items_metadata[
#     (dataset.items_metadata["category"] == "cat")
#     & (dataset.items_metadata["split"] == "test")
# ]
# start_time = time.time()
# cats_sketch = [dataset[row.id] for _, row in cats[:100].iterrows()]
# console.print(f"Loading 100 sketches in {time.time() - start_time}")
# del dataset, cats

# console.print()

console.print("3. Load from CISLAB CDN", style="green")

# tempfile can create a temporary directory
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    dataset = PhotoSketching(root=tmpdir, cislab_source=True)
    # show brief information of the dataset
    console.print(dataset)
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    console.print(f"Current memory usage: {memory_mb:.2f} MB")
    dogs = dataset.items_metadata[
        (dataset.items_metadata["category"] == "dog")
        & (dataset.items_metadata["split"] == "train")
    ]
    start_time = time.time()
    dogs_sketch = [dataset[row.id] for _, row in dogs[:100].iterrows()]
    console.print(f"Loading 100 sketches in {time.time() - start_time}")
    del dataset, dogs

    console.print()