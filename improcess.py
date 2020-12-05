import sys
import cv2 as cv
import numpy as np 
import tqdm

def opflow(first,second):
    gfirst = cv.cvtColor(first, cv.COLOR_RGB2GRAY)
    fsecond = cv.cvtColor(second, cv.COLOR_RGB2GRAY)
    hsv = np.zeros((first.shape))
    hsv[:,:,1] = cv.cvtColor(second, cv.COLOR_RGB2HSV)[:, :, 1]
    flow = cv.calcOpticalFlowFarneback(gfirst,gsecond, None, 0.5, 1, 15, 2, 5, 1.3, 0)
    r,theta = cv.cartToPolar(flow[...,0],flow[...,1])
    hsv[:,:,0] = theta*(180/np.pi/2)
    hsv[:,:,2] = cv.normalize(r,None,0,255,cv.NORM_MINMAX)
    hsv = np.asarray(hsv,dtype=np.float32)
    flowrgb = cv.cvtColor(hsv,cv.COLOR_HSV2RGB)
    return flowrgb

def drawlines(frame,lines):
    if(lines is None): return frame
    x,y,z = lines.shape
    for i in range(x):
        cv.line(frame, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 3, cv2.LINE_AA)
    return frame
            
def brighten(frame, brightness, contrast):
    if(brightness != 0):
        if(brightness>0):
            shade = brightness
            light = 255
        else:
            shade = 0
            light = 255 + brightness
        alpha = (light-shade)/255
        gamma = shade
        new = cv.addWeighted(frame,alpha,frame,0,gamma)
    else:
        new = frame.copy()

    if(contrast!=0):
        x = 131*(contrast+127)/(127*(131-contrast))
        alpha = x
        gamma = 127*(1-x)
        new = cv.addWeighted(new,alpha,new,0,gamma)
    return new

def threshold(frame):
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    low = np.array([20,100,100],dtype='uint8')
    high = np.array([30,255,255], dtype='uint8')
    y = cv.inRange(hsv,low,high)
    w = cv.inRange(gray,200,255)
    mask = cv.bitwise_or(w,y)
    new = cv.bitwise_and(gray,mask)
    return new

def drawflow(frame,flow,dist=16):
    h,w = frame.shape[:2]
    y,x = np.mgrid[dist/2:h:dist, dist/2:w:dist].reshape(2,-1).astype(int)
    flx,fly = flow[y,x].T
    lines = np.vstack([x,y,x+flx,y+fly]).T.reshape(-1,2,2)
    lines = np.int32(lines + 0.5)
    new = cv.cvtColor(frame,cv.COLOR_GRAY2BGR)
    cv.polylines(new,lines,False,(0,255,0))
    for (x1, y1), (x2, y2) in lines:
        cv.circle(new, (x1, y1), 1, (0, 255, 0), -1)
    return new

def removeDash(frame, top, bot):
    h,w = frame.shape[:2]
    return frame[top:bot,0:w]

def markLanes(frame):
    bright = brighten(frame,100,100)
    thresh = threshold(bright)
    gaussblur = cv.GaussianBlur(thresh,(5,5),0)
    edges = cv.Canny(gaussblur,100,200)
    h,w = edges.shape
    bL = [0,h-130]
    bR = [w,h-130]
    tL = [w/3+40,h/2]
    tR = [w/3*2-40,h/2]
    locs = np.array([bL,tL,tR,bR],np.int32)
    locs = locs.reshape((-1,1,2))
    black = np.zeros((h,w,1),np.uint8)
    poly = cv.fillPoly(black,[locs],(255,255,255))
    new = cv.bitwise_and(edges,edges,mask=poly)
    return new

def process(frame):
    bright = brighten(frame,30,15)
    gray = cv.cvtColor(bright, cv.COLOR_BGR2GRAY)
    h,w = gray.shape
    bL = [0,h-130]
    bR = [w,h-130]
    tL = [0,h/2+10]
    tR = [w,h/2+10]
    tC = [w/2,h/2-15]
    locs = np.array([bL,tL,tC,tR,bR],np.int32)
    locs = locs.reshape((-1,1,2))
    black = np.zeros((h,w,1),np.uint8)
    poly = cv.fillPoly(black,[locs],(255,255,255))
    roadMask = cv.bitwise_and(gray,gray,mask=poly)
    lanes = markLanes(frame)
    overlay = cv.add(roadMask,lanes)
    new = removeDash(roadMask,int(h/2-15),int(h-130))
    return new

#if __name__ == '__main__':
    #frame = cv.imread('./data/testrgb_imgs/27.jpg')
    #x = process(frame)


