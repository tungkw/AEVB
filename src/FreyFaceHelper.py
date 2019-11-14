import numpy as np
import scipy.io as sio
import os

class FreyFaceHelper:
    def __init__(self, data_path):
        full_path = os.path.join(data_path, "frey_rawface.mat")
        mat = sio.loadmat(full_path)
        print('loading file:', full_path)
        self.data = np.array(mat['ff'], dtype = np.uint8)
        self.data = np.transpose(self.data.reshape(28,20,-1),[2,0,1])
        print(self.data.shape)

