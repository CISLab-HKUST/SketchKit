from sketchkit.datasets import TUBerlin
from sketchkit.datasets import Sketchy

if __name__ == "__main__":
    dataset = TUBerlin()
    print(len(dataset))
    sketch = dataset[0]
    print(sketch.curve_num, sketch.path_num)

    # my_data2 = QuickDraw(category="cat", split="train")
    # print(len(my_data2))
    # print(my_data2[0])

    # data = Sketchy()
    # print(f"Total sketches: {len(data)}")
    # sketch = data[0]
    # print(sketch)
    # print(sketch.curve_num, sketch.path_num)
