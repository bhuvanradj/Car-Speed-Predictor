import cv2 as cv
import numpy as np 
import shutil
import tqdm
import itertools
import torch
import torchvision.transforms as T 

def frames(path):
    vid = cv.VideoCapture(path)
    ret,frame = vid.read()
    
    for x in itertools.count():
        ret,frame2 = vid.read()
        if(ret==False): break
        yield frame,frame2
        frame = frame2
    vid.release()
    cv.destroyAllWindows()

def brightness(img, k):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv[:,:,2] = hsv[:,:,2] * k
    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return rgb

def transform(img, k):
    img = brightness(img,k)
    img = cv.resize(img[100:440,:-90],(220,60),interpolation=cv.INTER_AREA)
    return img

def opflow(previous, nxt, k):
    previous = transform(previous,k)
    nxt = transform(nxt,k)
    pgray = cv.cvtColor(previous,cv.COLOR_BGR2GRAY)
    ngray = cv.cvtColor(nxt,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(previous)
    hsv[:,:,1] = 255
    flow = cv.calcOpticalFlowFarneback(pgray,ngray, None, 0.5, 1, 15, 3, 5, 1.5, 0)
    r,theta = cv.cartToPolar(flow[...,0],flow[...,1])
    hsv[:,:,0] = (theta*180)/(2*np.pi)
    hsv[:,:,2] = cv.normalize(r,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    return bgr

def gen(vid,txt,out):
    for n, (previous,nxt) in enumerate(tqdm.tqdm(frames(vid))):
        k = 0.2 + np.random.uniform()
        flow = opflow(previous,nxt,k)
        flowtensor = T.ToTensor()(flow).unsqueeze(0)
        if n == 0:
            topflow = flowtensor
        else: 
            topflow = torch.cat([topflow,flowtensor])
    speeds = np.loadtxt(txt)[1:]
    data = torch.utils.data.TensorDataset(topflow,torch.from_numpy(speeds).float())
    torch.save(data,out)
    
if __name__ == '__main__':
    gen('data/debug.mp4','data/debug.txt','output/debug.pt')