import numpy as np
import os

class MINSTHelper:
    def __init__(self, data_path):
        self.train_images = self.loadTrainImages(os.path.join(data_path, 'train-images.idx3-ubyte'))
        self.train_labels = self.loadTrainLabels(os.path.join(data_path, 'train-labels.idx1-ubyte'))
        self.test_images = self.loadTrainImages(os.path.join(data_path, 't10k-images.idx3-ubyte'))
        self.test_labels = self.loadTrainLabels(os.path.join(data_path, 't10k-labels.idx1-ubyte'))

    def loadTrainImages(self, data_path):
        f = open(data_path, 'rb')
        print('loading file:', data_path)

        magic_number = int.from_bytes(f.read(4), 'big')
        number_of_images = int.from_bytes(f.read(4), 'big')
        h = int.from_bytes(f.read(4), 'big')
        w = int.from_bytes(f.read(4), 'big')
        print('magic number', magic_number)
        print('number of images', number_of_images)
        print('number of rows', h)
        print('number of columns', w)

        byte_array = bytearray(f.read())
        images = np.array(byte_array, dtype=np.uint8).reshape(number_of_images, h, w)

        f.close()
        return images

    def loadTrainLabels(self, data_path):
        f = open(data_path, 'rb')
        print('loading file:', data_path)
            
        magic_number = int.from_bytes(f.read(4), 'big')
        number_of_items = int.from_bytes(f.read(4), 'big')
        print('magic number', magic_number)
        print('number of items', number_of_items)

        byte_array = bytearray(f.read())
        images = np.array(byte_array, dtype=np.uint8).reshape(number_of_items, 1)

        f.close()
        return images

