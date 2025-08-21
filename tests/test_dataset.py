from sketchkit.datasets import TUBerlin, QuickDraw

if __name__ == '__main__':
    my_data = TUBerlin()
    print(len(my_data))
    print(my_data[0])

    my_data2 = QuickDraw(category="cat", split="train")
    print(len(my_data2))
    print(my_data2[0])