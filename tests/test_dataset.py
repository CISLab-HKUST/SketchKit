from sketchkit.datasets import TUBerlin
from sketchkit.datasets import SketchyDatabase

if __name__ == "__main__":
    # dataset = TUBerlin()
    # print(len(dataset))
    # sketch = dataset[0]
    # print(sketch.stroke_num, sketch.path_num)

    # my_data2 = QuickDraw(category="cat", split="train")
    # print(len(my_data2))
    # print(my_data2[0])

    data = SketchyDatabase()
    print(f"Total sketches: {len(data)}")
    sketch = data[0]
    print(sketch.stroke_num, sketch.path_num)
    print("Loaded first sketch successfully.")
