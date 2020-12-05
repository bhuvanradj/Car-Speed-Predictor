import os
import sys
import numpy as np
import imageio
import skimage
import cv2 as cv


def overlay():
    font = cv.FONT_HERSHEY_DUPLEX
    fontScale = 1.0
    fontColor1 = (55,155,255)
    fontColor2 = (52,255,52)
    lineType = 2
    
    h = 480
    w = 640
    print('Completed Loading\n')
    
    predfile = open('./output/test.txt')
    #actualfile = open('./data/train.txt')
    preds = predfile.readlines()
    #actual = actualfile.readlines()
    n = len(preds)
    testoutput = cv.VideoWriter('./output/sub.mp4',cv.VideoWriter_fourcc(*'mp4v'),20,(w,h))

    print('Overlaying...\n')
    for i in range(n):
        sys.stdout.write("\rOn frame %d of video" % (i))
        rgb = str(i) + '.jpg'
        r = cv.imread('./data/testrgb_imgs/' + rgb)
        cv.putText(r,"Predicted: " + preds[i].strip('[').split(']')[0],
            (50,50),
            font,
            fontScale,
            fontColor1,
            lineType)
        '''
        cv.putText(r,"Actual: " + actual[i].split('\n')[0],
            (440,50),
            font,
            fontScale,
            fontColor2,
            lineType)'''
        #cv.imshow('frame',r)
        #k = cv.waitKey(0);
        testoutput.write(r)
    predfile.close()
    #actualfile.close()
    testoutput.release()
    print('\nTest Output Video Written!')

if __name__ == '__main__':
    overlay()