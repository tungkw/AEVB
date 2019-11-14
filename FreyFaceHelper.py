import numpy as np
import scipy.io as sio

class FreyFaceHelper:
    def __init__(self, data_path):
        full_path = data_path + "frey_rawface.mat"
        mat = sio.loadmat(full_path)
        print('loading file:', full_path)
        self.data = np.array(mat['ff'], dtype = np.uint8)
        self.data = np.transpose(self.data.reshape(28,20,-1),[2,0,1])
        print(self.data.shape)

if __name__ == "__main__":
    import setting
    import cv2
    freyface_helper = FreyFaceHelper(setting.freyface_path)
    for i in range(len(freyface_helper.data)):
        cv2.imshow(None, freyface_helper.data[i])
        cv2.waitKey(50000)