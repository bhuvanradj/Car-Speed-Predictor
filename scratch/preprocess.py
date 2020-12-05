import sys
import cv2 as cv
import numpy as np 
import scipy
import tqdm

def opflow(one,two):
    hsv = np.zeros_like(one)
    one = cv.cvtColor(one,cv.COLOR_BGR2GRAY)
    two = cv.cvtColor(two,cv.COLOR_BGR2GRAY)
    hsv[...,1] = 255
    flow = cv.calcOpticalFlowFarneback(one,two, None, 0.5, 1, 12, 2, 7, 1.4, 0)
    r,theta = cv.cartToPolar(flow[...,0],flow[...,1])
    hsv[...,0] = theta*180/np.pi/2
    hsv[...,2] = cv.normalize(r,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    return bgr

def flowvideo(path):
    vid = cv.VideoCapture(path)
    ret,one = vid.read()
    print('Loading Video')
    ct=0
    while ret:
        ret,two = vid.read()
        if not ret: break
        flow = opflow(one,two)
        cv.imwrite('./data/testflow_imgs/' + str(ct) + '.jpg', flow)
        cv.imwrite('./data/testrgb_imgs/' + str(ct) + '.jpg', two)
        sys.stdout.write("\rOn frame %d of video" % ct)
        one = two
        ct+=1
    print('\n')
    print(str(ct) + ' frames stored')
    vid.release()



if __name__ == '__main__':
    flowvideo('./data/test.mp4')
    
