import psutil
from rich.console import Console
import time
import tempfile

from sketchkit.datasets import PhotoSketching
from sketchkit.datasets import QuickDraw

if __name__ == "__main__":
    console = Console()

    console.print("QuickDraw Dataset Test", style="bold blue")
    console.print()

    console.print("1. Default Load", style="green")
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = QuickDraw(
            root=tmpdir
        )  # default load, which loads metadata and sketch data on demand
        # show brief information of the dataset
        console.print(dataset)
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        console.print(f"Current memory usage: {memory_mb:.2f} MB")

        # search data with "category = cat" and "split = train"
        cats = dataset.items_metadata[
            (dataset.items_metadata["category"] == "cat")
            & (dataset.items_metadata["split"] == "train")
        ]
        start_time = time.time()
        cats_sketch = [dataset[row.id] for _, row in cats[:100].iterrows()]
        console.print(f"Loading 100 sketches in {time.time() - start_time}")

        del dataset, cats
        console.print()

    console.print("2. Load All", style="green")
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = QuickDraw(root=tmpdir, load_all=True)
        # show brief information of the dataset
        console.print(dataset)
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        console.print(f"Current memory usage: {memory_mb:.2f} MB")
        # search data with "category = cat" and "split = test"
        cats = dataset.items_metadata[
            (dataset.items_metadata["category"] == "cat")
            & (dataset.items_metadata["split"] == "test")
        ]
        start_time = time.time()
        cats_sketch = [dataset[row.id] for _, row in cats[:100].iterrows()]
        console.print(f"Loading 100 sketches in {time.time() - start_time}")
        del dataset, cats

        console.print()

    console.print("Load from CISLAB CDN", style="green")
    console.print("PhotoSketching Dataset Test", style="bold blue")

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset = PhotoSketching(root=tmpdir, cislab_source=True)
        # show brief information of the dataset
        console.print(dataset)
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        console.print(f"Current memory usage: {memory_mb:.2f} MB")

        start_time = time.time()
        sketches = [dataset[i] for i in range(min(5, len(dataset)))]
        console.print(
            f"Loading {len(sketches)} sketches in {time.time() - start_time:.2f}s"
        )
        del dataset, sketches

        console.print()
