import os
import sys
import numpy as np
import imageio
import skimage
import cv2 as cv
import keras
from improcess import *
from CNNv2 import Nvidia_CNN

def retspeed():
    speedfile = open('./data/train.txt')
    strspeed = speedfile.readlines()
    speeds = []
    for speed in strspeed:
        speed = speed.strip('\n')
        speeds.append(float(speed))
    speedfile.close()
    return speeds

def eval(): 
    path = './data/train.mp4'
    vid = cv.VideoCapture(path)
    print('Loading Video')
    n = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
    speeds = retspeed()

    valsize = 0.2
    vald = int(n*valsize)
    valg = int(n/vald)
    
    framebatch = []
    speedbatch = []
    frameval = []
    speedeval = []
    ct=0
    f=0
    while (ct < (n-1)):
        ret,new = vid.read()
        roi = process(new)
        if ct==0:
            oroi = roi
            flow = cv.calcOpticalFlowFarneback(oroi,roi, None, 0.5, 1, 15, 2, 5, 1.3, 0)
        else:
            flow = cv.calcOpticalFlowFarneback(oroi,roi, None, 0.5, 1, 15, 2, 5, 1.3, 0)
        if(f==valg):
            f=0
            frameval.append(flow)
            speedeval.append(speeds[ct])
        oroi = roi
        ct+=1
        f+=1
        cv.imshow('frame', drawflow(roi,flow))
        cv.waitKey(1)
        sys.stdout.write("\rOn frame %d of video" % ct)
    print('\n')
    
    etmp = list(zip(frameval,speedeval))
    np.random.shuffle(etmp)
    frameval,speedeval = zip(*etmp)

    vid.release()
    cv.destroyAllWindows()

    X = np.array(frameval)
    Y = np.array(speedeval)
    model = keras.models.load_model('./models/modelv4.h5')
    metrics = model.evaluate(X,Y,verbose=1)
    print('Metrics:')
    print(metrics)

def train():
    path = './data/train.mp4'
    vid = cv.VideoCapture(path)
    print('Loading Video')
    n = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
    speeds = retspeed()

    valsize = 0.15
    vald = int(n*valsize)
    valg = int(n/vald)
    
    framebatch = []
    speedbatch = []
    frameval = []
    speedeval = []
    ct=0
    f=0
    while (ct < (n-1)):
        ret,new = vid.read()
        roi = process(new)
        if ct==0:
            oroi = roi
            flow = cv.calcOpticalFlowFarneback(oroi,roi, None, 0.5, 1, 15, 2, 5, 1.3, 0)
            fshape = flow.shape
            model = Nvidia_CNN(fshape)
        else:
            flow = cv.calcOpticalFlowFarneback(oroi,roi, None, 0.5, 1, 15, 2, 5, 1.3, 0)
        if(f==valg):
            f=0
            frameval.append(flow)
            speedeval.append(speeds[ct])
        else:
            framebatch.append(flow)
            speedbatch.append(speeds[ct])
        oroi = roi
        ct+=1
        f+=1
        cv.imshow('frame', drawflow(roi,flow))
        cv.waitKey(1)
        sys.stdout.write("\rOn frame %d of video" % ct)
    print('\n')
    
    tmp = list(zip(framebatch,speedbatch))
    np.random.shuffle(tmp)
    framebatch,speedbatch = zip(*tmp)

    etmp = list(zip(frameval,speedeval))
    np.random.shuffle(etmp)
    frameval,speedeval = zip(*etmp)

    vid.release()
    cv.destroyAllWindows()

    print('Training...\n')
    n = len(speedbatch)
    trainframe = []
    trainspeed = []
    i=0
    f=0
    bsize=600
    while(i<n):
        trainframe.append(framebatch[i])
        trainspeed.append(speedbatch[i])
        i+=1
        f+=1
        if(f==bsize or i==(n-1)):
            sys.stdout.write("\rOn frame %d of video" % i)
            X = np.array(trainframe)
            Y = np.array(trainspeed)
            model.fit(x=X, y=Y, verbose=1, epochs=15, batch_size=32, shuffle=True)
            f=0
            trainframe=[]
            trainspeed=[]
    model.save('./models/modelv4.h5')
    print('\nTraining Complete!\n')
    eval(frameval,speedeval)

def test():
    model = keras.models.load_model('./models/modelv4.h5')
    path = './data/train.mp4'
    vid = cv.VideoCapture(path)
    print('Loading Video')
    n = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
    text = open('./output/modelv4train.txt','w')
    oldspeed=None
    ct=0
    predframe = [0]
    while (ct < (n-2)):
        sys.stdout.write("\rOn frame %d of video" % ct)
        ret,new = vid.read()
        roi = process(new)
        if ct==0:
            oroi = roi
            flow = cv.calcOpticalFlowFarneback(oroi,roi, None, 0.5, 1, 15, 2, 5, 1.3, 0)
        else:
            flow = cv.calcOpticalFlowFarneback(oroi,roi, None, 0.5, 1, 15, 2, 5, 1.3, 0)
        predframe[0] = flow
        X = np.array(predframe)
        pred = model.predict(X)
        curspeed = pred
        if(oldspeed is None): oldspeed = curspeed
        curspeed = (curspeed+oldspeed)/2
        oldspeed = curspeed
        text.write('%s\n' % curspeed)
        cv.putText(new, str(curspeed), (50, 50), cv.FONT_HERSHEY_DUPLEX, 1.0, color=(51, 153, 255),lineType=2)
        oroi = roi
        ct+=1
        #cv.imshow('frame', new)
        #cv.waitKey(1)
    text.close()
    vid.release()
    cv.destroyAllWindows()
    print('\nPredictions Done!')


if __name__ == '__main__':
    #train()
    test()
    #eval()
    