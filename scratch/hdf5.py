import h5py
import tqdm
import pandas as pd 
import numpy as np 
import cv2 as cv 
from skimage.transform import resize 
import matplotlib.pyplot as plt

with open('data/train.txt','r') as f:
    tmp = f.read()
speeds = [float(i) for i in tmp.split('\n')]
shape = len(speeds)
path = 'data/train.hdf5'
hdf5_data = h5py.File(path, 'w')
hdf5_data.create_dataset('speed', shape=(shape,1), maxshape=(shape,1))
hdf5_data.create_dataset('frame', shape=(shape,224,224,3), maxshape=(shape,224,224,3),chunks=(1,224,224,3))
hdf5_data.create_dataset('flow', shape=(shape,224,224,3), maxshape=(shape,224,224,3),chunks=(1,224,224,3))

def setup():
    ct = 0
    cap = cv.VideoCapture('data/train.mp4')
    ret,tmp = cap.read()
    frame = tmp[160:360, :]
    previous = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[...,1] = 255
    while ret:
        if(ct%100==0 and ct >0):
            print(ct)
            plt.imshow(fr)
            plt.show()
        ret,tmp = cap.read()
        if(ret):
            fr = tmp[160:360, :]
            nxt = cv.cvtColor(fr,cv.COLOR_BGR2GRAY)
            opflow = cv.calcOpticalFlowFarneback(previous,nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            r,theta = cv.cartToPolar(opflow[...,0],opflow[...,1])
            hsv[...,0] = (theta*180)/(2*np.pi)
            hsv[...,2] = cv.normalize(r,None,0,255,cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
            ct+=1
            spd = speeds[ct]
            fr = resize(fr,((224,224,3)))
            hdf5_data['frame'][ct,:,:,:] = fr
            hdf5_data['speed'][ct,:] = spd
            bgr = resize(bgr,((224,224,3)))
            hdf5_data['flow'][ct,:,:,:] = bgr
            previous == nxt

#setup()
hdf5_data.close()



            



    

