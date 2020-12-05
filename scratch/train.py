import os
import sys
import re
import numpy as np
import imageio
import skimage
import cv2 as cv
import keras
from CNNv2 import Nvidia_CNN

def process(path):
    img = imageio.imread('./data/flow_imgs/' + path)[200:400]
    img = skimage.transform.resize(img, [66,200]) / 255
    return img

def load():
    print('Loading Frames\n')
    images=[]
    speeds=[]
    spds = np.loadtxt('./data/train.txt')
    fs = os.listdir('./data/flow_imgs/')
    for f in fs:
        if(f.split('.')[1] != 'jpg'):
            fs.remove(f)
    imgs = sorted(fs, key=lambda x: int(x.split('.')[0]))
    n = len(imgs)
    ct=0
    for i in range(n-1):
        sys.stdout.write("\rOn frame %d of video" % ct)
        path = imgs[i]
        img = process(path)
        images.append(img)
        speeds.append((spds[i]+spds[i+1])/2)
        ct+=1
    return np.asarray(images),np.asarray(speeds)

def train():
    imgs,speeds = load()
    print('\nCompleted Loading')
    model = Nvidia_CNN(imgs.shape)
    print('\nTraining Model')
    model.fit(imgs[0,:,:,:,:],speeds,batch_size=32,epochs=40,verbose=1,validation_split=0.2,shuffle=True)
    model.save('./models/v3p2.h5')
    print('\nModel Saved')

def test():
    imgs = load()
    print('Completed Loading\n')
    print('Testing Model\n')
    text = open('./output/actualtest.txt','w')
    model = keras.models.load_model('./models/secondmodel.h5')
    ct=1
    for img in imgs:
        sys.stdout.write("\rOn frame %d of video" % ct)
        prediction = model.predict(np.expand_dims(img,axis=0))[0][0]
        text.write('%s\n' % prediction)
        ct+=1
    text.close()
    print('Text File Written\n')
    #metrics = model.evaluate(imgs,speeds,batch_size=32)
    #print('Metrics:')
    #rint(metrics)

def eval():
    predf = open('./output/modelv4train.txt','r')
    actualf = open('./data/train.txt','r')
    preds=predf.readlines()
    actuals=actualf.readlines()
    meansquared = 0
    
    for i in range(len(preds)):
        actual = float(actuals[i].split('\n')[0])
        pred = float(preds[i].strip('[').split(']')[0])
        meansquared += (actual-pred)**2
    mse = meansquared/len(preds)
    print("MSE = " + str(mse))
    
    predf.close()
    actualf.close()


if __name__ == '__main__':
    #train()
    #test()
    eval()