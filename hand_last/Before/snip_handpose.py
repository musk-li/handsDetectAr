#coding=utf-8
import time
import torch
import cv2
from PIL import Image
import cvzone
import numpy as np
from numpy import *
from cvzone.HandTrackingModule import HandDetector
from yolo import YOLO


# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)  # width
cap.set(4, 720)   # height
detector = HandDetector(detectionCon=0.8, maxHands=1)  # confidence and the num of hands
colorR = (255, 0, 255)
#lmList = []
cx, cy, w, h = 100, 100, 200, 200

yolo = YOLO()

class DeagRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor

rectList = []
for x in range(5):
    rectList.append(DeagRect([x * 250 + 150, 150]))

# Variables
delayCounter = 0

# Loop
while True:
    # Get image from webcam (frame)
    success, img = cap.read()
    img = cv2.flip(img, 1)  # 1 flip horizontally

    # Detection of hand
    allHands, img = detector.findHands(img, flipType=False)
    #snip_img = zeros((200,200))

    # Check for Hand
    if len(allHands) != 0:
        lmList = allHands[0]['lmList']
        cursor = lmList[8]  # 食指的中心点坐标
        l, _, _ = detector.findDistance(lmList[8], lmList[12], img)   # 返回食指和中指的距离
        m_b, _, _ = detector.findDistance(lmList[4], lmList[8], img)  # 返回食指和拇指的距离
        # 触发检测框
        if l < 50:
            if cursor[0] > 200 and cursor[1] > 200:
                L_up_row = cursor[1]-200
                L_up_con = cursor[0]-200
                row_R = cursor[1]
                con_R = cursor[0]
                img[L_up_row:row_R, con_R] = [255, 255, 255]
                img[row_R, L_up_con:con_R] = [255, 255, 255]
                img[L_up_row:row_R, L_up_con] = [255, 255, 255]
                img[L_up_row, L_up_con:con_R] = [255, 255, 255]
                snip_img = img[L_up_row:row_R, L_up_con:con_R]
        # 触发开始检测
                if m_b < 50 and delayCounter == 0:
                    img[L_up_row:row_R, con_R] = [255, 0, 0]
                    img[row_R, L_up_con:con_R] = [255, 0, 0]
                    img[L_up_row:row_R, L_up_con] = [255, 0, 0]
                    img[L_up_row, L_up_con:con_R] = [255, 0, 0]
                    #time.sleep(0.2)
                    cv2.imwrite('snip.jpg', snip_img)

                    imag1 = Image.open('snip.jpg')
                    r_image, out_class = yolo.detect_image(imag1)

                    cv2.putText(img, str(out_class), (L_up_con, L_up_row), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 0, 255), 2)  # 照片/添加的文字/左下角坐标/字体/字体大小/颜色/字体粗细
                    delayCounter = 1
                    #time.sleep(0.2)
                    print(out_class)
            else:
                cv2.putText(img, str('out of the range'), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 0, 255), 2)

            for rect in rectList:
                rect.update(cursor)
    # 避免误触发
    if delayCounter != 0:
        delayCounter += 1
        if delayCounter > 10:
            delayCounter = 0

    imgNew = np.zeros_like(img, np.uint8)

    """
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), colorR, cv2.FILLED)
        #cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20,rt=0)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20)
    """

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    key = cv2.waitKey(1)
    if key == ord('q'):  # quit
        break
    pass
