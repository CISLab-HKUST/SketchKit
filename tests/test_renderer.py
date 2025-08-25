from sketchkit.datasets import TUBerlin #, QuickDraw

if __name__ == '__main__':
    my_data = TUBerlin()
    print(len(my_data))
    print(my_data[0][0].rasterize('output.png'))