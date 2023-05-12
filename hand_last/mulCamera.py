import cv2
from cvzone.HandTrackingModule2 import HandDetector2
import numpy as np


cap1 = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap1.set(3, 800)      # width 1280
cap1.set(4, 600)
cap2 = cv2.VideoCapture(1,cv2.CAP_DSHOW)
cap2.set(3, 800)      # width 1280
cap2.set(4, 600)
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector = HandDetector2(detectionCon=0.8, maxHands=2)
while True:
    suc1,img1 = cap1.read()
    suc2,img2 = cap2.read()
    # img1 = cv2.flip(img1, 1)
    # img2 = cv2.flip(img2, 1)
    img1 = cv2.resize(img1,(int(width),int(height)),interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)

    allHands,img1, img2 = detector.findHands(img1,img2, draw=True, flipType=True)

    if len(allHands) != 0 :
        if len(allHands)==2:
            if allHands[0]['type'] == 'Left':
                lmListLeft = allHands[0]['lmList']
                lmListRight = allHands[1]['lmList']
            else:
                lmListLeft = allHands[1]['lmList']
                lmListRight = allHands[0]['lmList']
        else:
            lmListRight = allHands[0]['lmList']
        cursor = lmListRight[8]
        cv2.rectangle(img2, (cursor[0]-150, cursor[1]-150),(cursor[0],cursor[1]), (255, 255, 255), 2)
    # dst = cv2.addWeighted(img1,0.5,img2,0.5,0)

    dst = np.hstack((img1,img2))
    cv2.imshow('Fuse',dst)
    # cv2.imshow('img1',img1)
    # cv2.imshow('img2',img2)
    cv2.waitKey(10)
    # key = cv2.waitKey(10)
    # if int(key) == 113:
    #     break
