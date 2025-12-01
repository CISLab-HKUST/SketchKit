from sketchkit.cleanup import Cleaner


if __name__ == '__main__':
    cleaner = Cleaner(method="MasterSketch")

    img = "01.png"
    sketch = cleaner.run(img)
