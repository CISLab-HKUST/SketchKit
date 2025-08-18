from sketchkit.datasets import TUBerlin

if __name__ == '__main__':
    my_data = TUBerlin()

    print(len(my_data))
    print(my_data[0])