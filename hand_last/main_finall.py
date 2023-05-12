#coding=utf-8
import cv2
from PIL import Image
import cvzone
from numpy import *
from cvzone.HandTrackingModule1 import HandDetector
import numpy as np
from MethodUse import readtxt,cv2ImgAddText,fingerStill,inArea,distance
import time
from yolo import YOLO
import sys

# Set parameters
lmList = []
w, h = 150, 150       # the size of main menu
colorR = (0, 0, 255)  # the color of main menu
mode = ''             # initialize mode

yolo = YOLO()         # initialize detection

# Variables of Keyboard
myEquation = ''
delayCounter = 0

# Webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 2280)      # width 1280
cap.set(4, 900)       # height 720
detector = HandDetector(detectionCon=0.8, maxHands=2)  # confidence and the num of hands
center = [410,400]
wHalf = 310
hHalf = 300
pTime = 0
Storedata = []
Storedata0 = []
Storedata1 = []
Storedata2 = []
Storedata3 = []
Storedata4 = []
clickopen = False
fourclick = False
frontInter = [15,20,25,35]
frontSize = [10,15,20,30]
FrontInter = frontInter[2]
FrontSize = frontSize[2]
okkk = 0
class BackgroundRect():
    def __init__(self, center,w,h):
        self.center = center
        self.w = w
        self.h = h
    def update(self,cursor):
        cx, cy = self.center
        w=self.w
        h = self.h
        if cx - w  < cursor[0] < cx +w and \
                cy - h  < cursor[1] < cy + h :
            self.center = cursor
    def updateSize(self,whalf,hhalf):
        self.w = whalf
        self.h = hhalf
rec = BackgroundRect(center,wHalf,hHalf)
# Draw main menu rectangle
class DeagRect():
    def __init__(self, posCenter, size=[w, h]):
        self.posCenter = posCenter
        self.size = size

rectList = []
for x in range(4):  # to generate two rectangles
    rectList.append(DeagRect([x * 250 + 50, 50]))  # left_top (50,50) (300,50)

# Loop
while True:
    # Get image from webcam (frame)
    success, img = cap.read()
    img = cv2.flip(img, 1)  # 1 flip horizontally / 0 vertically

    # Detection of hand
    allHands, img = detector.findHands(img, draw=True, flipType=False)  # if u flip the img, set flipType=False

    # Choose mode (Entertainment: target recognition Office: calculator)
    # by detect the distance of index finger and middle finger
    if len(allHands) != 0:
        lmList = allHands[0]['lmList']
        cursor = lmList[8]  # the location of index finger cursor[0]: x  cursor[1]: y
        l, _, _ = detector.findDistance(lmList[8], lmList[12], img)
        if l < 50:
            # Choose mode
            if 50 < cursor[0] < 50 + w and 100 < cursor[1] < 100 + h:
                colorR = [255, 0, 0]
                mode = 'detect_pic'  # Entertainment: Target recognition
            elif 300 < cursor[0] < 300 + w and 100 < cursor[1] < 100 + h:
                colorR = [0, 255, 255]
                mode = 'calculate'   # Office: Calculator
            elif 550 < cursor[0] < 550 + w and 100 < cursor[1] < 100 + h:
                colorR = [0, 255, 255]
                mode = 'bigsmall'
            elif 800 < cursor[0] < 800 + w and 100 < cursor[1] < 100 + h:
                colorR = [0, 255, 255]
                mode = 'trackobject'
                detecT = False
                clickt = False

                # Entertainment: Target recognition
    if mode == "detect_pic":
        # Check for Hand
        if len(allHands) != 0:
            lmList = allHands[0]['lmList']
            cursor = lmList[8]  # the location of index finger
            l, _, _ = detector.findDistance(lmList[8],lmList[12],img)    # return the distance of index and middle finger
            m_b, _, _ = detector.findDistance(lmList[4],lmList[8], img)  # return the distance of index finger and thumb
            # trigger detection box
            if l < 50:
                win_size = 416  # the size of detection box
                if cursor[0] > win_size and cursor[1] > win_size:  # border condition
                    box_left = cursor[0] - win_size
                    box_top = cursor[1] - win_size
                    cv2.rectangle(img, (box_left, box_top), cursor, (0, 255, 0), 2)         # cv2.rectangle(img,pt1,pt2,color,thickness) 边框
                    snip_img = img[box_top:cursor[1], box_left:cursor[0]]                   # capture pic (cv2 BGR)
                    # start detection
                    if m_b < 50:# and delayCounter == 0:
                        cv2.rectangle(img, (box_left, box_top), cursor, (255, 255, 0), 2)   # cv2.rectangle(img,pt1,pt2,color,thickness) 边框
                        cv2.imwrite('snip.jpg', snip_img)
                        img_temp = Image.open('snip.jpg')
                        res_image, out_class = yolo.detect_image(img_temp)
                        # res_image.show()
                        cv2.putText(img, str(out_class), (box_left, box_top), cv2.FONT_HERSHEY_PLAIN,
                                    2, (0, 0, 255), 2)                                      # 照片/添加的文字/左下角坐标/字体/字体大小/颜色/字体粗细
                        # delayCounter = 1
                        # print(out_class)
                else:
                    cv2.putText(img, str('out of the range'), (5, 30), cv2.FONT_HERSHEY_PLAIN,  # 超边界提示
                                2, (0, 0, 255), 2)

        # # Avoid false trigger
        # if delayCounter != 0:
        #     delayCounter += 1
        #     if delayCounter > 3:
        #         delayCounter = 0

        # # Display image
        # cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):  # quit
            break
        pass
    if mode == "calculate":
        class Button:
            def __init__(self, pos, width, height, value):
                self.pos = pos
                self.width = width
                self.height = height
                self.value = value

            def draw(self, img):  # draw calculator
                cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                              (225, 225, 225), cv2.FILLED)  # cv2.rectangle(img,pt1,pt2,color,thickness)
                cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                              (50, 50, 50), 3)   # cv2.rectangle(img,pt1,pt2,color,thickness) 边框

                cv2.putText(img, self.value, (self.pos[0] + 40, self.pos[1] + 60), cv2.FONT_HERSHEY_PLAIN,
                            2, (50, 50, 50), 2)  # 照片/添加的文字/左下角坐标/字体/字体大小/颜色/字体粗细

            def checkClick(self, x, y):  # confirm click
                if self.pos[0] < x < self.pos[0] + self.width and \
                        self.pos[1] < y < self.pos[1] + self.height:
                    cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                                  (255, 255, 255), cv2.FILLED)  # cv2.rectangle(img,pt1,pt2,color,thickness)
                    cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                                  (50, 50, 50), 3)  # cv2.rectangle(img,pt1,pt2,color,thickness) 边框

                    cv2.putText(img, self.value, (self.pos[0] + 25, self.pos[1] + 80), cv2.FONT_HERSHEY_PLAIN,
                                5, (0, 0, 0), 5)  # 照片/添加的文字/左下角坐标/字体/字体大小/颜色/字体粗细
                    return True
                else:
                    return False

        # # Webcam
        # detector = HandDetector(detectionCon=0.8, maxHands=1)  # set confidence and the num of hands

        # Creating Buttons
        buttonListValues = [['7', '8', '9', '*'],
                            ['4', '5', '6', '-'],
                            ['1', '2', '3', '+'],
                            ['0', '/', '.', '=']]
        buttonList = []
        for x in range(4):
            for y in range(4):
                xpos = x * 100 + 800  # the size of each button 100*100
                ypos = y * 100 + 150
                buttonList.append(Button((xpos, ypos), 100, 100, buttonListValues[y][x]))

        # Draw all buttons
        cv2.rectangle(img, (800, 50), (800 + 400, 70 + 100),
                      (225, 225, 225), cv2.FILLED)  # cv2.rectangle(img,pt1,pt2,color,thickness)
        cv2.rectangle(img, (800, 50), (800 + 400, 70 + 100),
                      (50, 50, 50), 3)              # cv2.rectangle(img,pt1,pt2,color,thickness) edge

        for button in buttonList:
            button.draw(img)

        # Check for Hand
        if allHands:
            # Find distance between fingers
            lmList = allHands[0]['lmList']
            length, _, img = detector.findDistance(lmList[8], lmList[12], img)
            #print(length)
            x, y = lmList[8]

            # If clicked check which button and perform action
            if length < 50 and delayCounter == 0:
                for i, button in enumerate(buttonList):  # enumerate 索引序列 索引+数据
                    if button.checkClick(x, y):
                        myValue = buttonListValues[int(i % 4)][int(i / 4)]  # column row
                        if myValue == "=":
                            myEquation = str(eval(myEquation))
                        else:
                            myEquation += myValue
                        delayCounter = 1
                        # time.sleep(0.1) # perform a bad effect

        # to Avoid duplicates
        if delayCounter != 0:
            delayCounter += 1
            if delayCounter > 10:
                delayCounter = 0

        # Display the Equation/Result
        cv2.putText(img, myEquation, (810, 120), cv2.FONT_HERSHEY_PLAIN,
                    3, (50, 50, 50), 3)  # 照片/添加的文字/左下角坐标/字体/字体大小/颜色/字体粗细
        key = cv2.waitKey(1)
        if key == ord('c'):    # clear
            myEquation = ''
        elif key == ord('q'):  # quit
            break
        pass
    if mode == "bigsmall":
        RecLeftup = (rec.center[0] - rec.w, rec.center[1] - rec.h)
        RecRightdown = (rec.center[0] + rec.w, rec.center[1] + rec.h)

        if len(allHands) != 0:
            if len(allHands) == 2:
                if allHands[0]['type'] == 'Left':
                    lmListLeft = allHands[0]['lmList']
                    lmListRight = allHands[1]['lmList']
                else:
                    lmListLeft = allHands[1]['lmList']
                    lmListRight = allHands[0]['lmList']
                # l, _, _ = detector.findDistance(lmListLeft[8], lmListRight[8], img)
                # b, _, _ = detector.findDistance(lmListLeft[4], lmListRight[4], img)
                Lthumbfin = lmListLeft[4]
                Lindexfin = lmListLeft[8]
                Rthumbfin = lmListRight[4]
                Rindexfin = lmListRight[8]
                Lthumbfinger = inArea(lmListLeft[4], RecLeftup, RecRightdown)
                Lindexfinger = inArea(lmListLeft[8], RecLeftup, RecRightdown)
                Rthumbfinger = inArea(lmListRight[4], RecLeftup, RecRightdown)
                Rindexfinger = inArea(lmListRight[8], RecLeftup, RecRightdown)
                Storedata1, clickLt = fingerStill(Lthumbfin, Storedata1, count=15)
                Storedata2, clickLi = fingerStill(Lindexfin, Storedata2, count=15)
                Storedata3, clickRt = fingerStill(Rthumbfin, Storedata3, count=15)
                Storedata4, clickRi = fingerStill(Rindexfin, Storedata4, count=15)
                if clickLt and clickLi and clickRt and clickRi:
                    fourclick = not fourclick

                if Lthumbfinger and Rthumbfinger and Lindexfinger and Rindexfinger and not fourclick:
                    whalf = int(distance(Lindexfin, Rindexfin))
                    hhalf = int(distance(Rthumbfin, Rindexfin))
                    # print(whalf,hhalf)
                    if 250 <= whalf <= 800 and 250 <= hhalf <= 800:
                        if 250 <= whalf <= 300 and 250 <= hhalf <= 300:
                            FrontInter = frontInter[1]
                            FrontSize = frontSize[1]
                        elif 300 < whalf <= 450 and 300 < hhalf <= 450:
                            FrontInter = frontInter[2]
                            FrontSize = frontSize[2]
                        elif 450 < whalf <= 800 and 450 < hhalf <= 800:
                            FrontInter = frontInter[3]
                            FrontSize = frontSize[3]
                        rec.updateSize(whalf, hhalf)
                    elif hhalf < 200 and whalf < 200:
                        FrontInter = frontInter[0]
                        FrontSize = frontSize[0]
                        whalf = 180
                        hhalf = 180
                        rec.updateSize(whalf, hhalf)

            else:
                lmList = allHands[0]['lmList']
                finger8 = lmList[8]
                Indesfinger = inArea(lmList[8], RecLeftup, RecRightdown)
                Middlefinger = inArea(lmList[12], RecLeftup, RecRightdown)
                Ringfinger = inArea(lmList[16], RecLeftup, RecRightdown)
                if inArea(lmList[8], RecLeftup, RecRightdown):
                    Storedata, click = fingerStill(finger8, Storedata, count=7)
                    if click: clickopen = not clickopen
                if Indesfinger and Middlefinger and Ringfinger and clickopen:
                    rec.update(finger8)
                    cv2.circle(img, (50, 50), 15, (0, 255, 0), cv2.FILLED)
                if not clickopen:
                    cv2.circle(img, (50, 50), 15, (0, 0, 255), cv2.FILLED)

        xi = RecLeftup[0]
        yi = RecLeftup[1] - 25
        blk = np.zeros(img.shape, np.uint8)
        cv2.rectangle(blk, RecLeftup, RecRightdown, (255, 255, 255), -1)
        img = cv2.addWeighted(img, 1.0, blk, 0.5, 1)

        if len(allHands) == 2 and Lthumbfinger and Rthumbfinger and Lindexfinger and Rindexfinger:
            if not fourclick:
                cv2.circle(img, Lthumbfin, 15, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, Lindexfin, 15, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, Rthumbfin, 15, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, Rindexfin, 15, (0, 255, 0), cv2.FILLED)
            else:
                cv2.circle(img, Lthumbfin, 15, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, Lindexfin, 15, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, Rthumbfin, 15, (0, 0, 255), cv2.FILLED)
                cv2.circle(img, Rindexfin, 15, (0, 0, 255), cv2.FILLED)
        text_line = readtxt('ReadChinese.txt', 30)
        readF = text_line[0]
            # +text_line[1]+text_line[2]+text_line[3]
        for i, text in enumerate(readF):
            if text:
                # cv2.putText(img, text, (xi, yi + 40 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                #             1, (0, 0, 255), 1)

                img = cv2ImgAddText(img, text, xi, yi + FrontInter * (i + 1) + 15, (0, 0, 139), FrontSize)
    if mode == "trackobject":
        cv2.putText(img, 'trackobject', (700 + 50, 50 + 105), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,0), 3)
        if len(allHands) != 0:
            lmList = allHands[0]['lmList']
            cursor = lmList[8]
            Storedata0, clickt = fingerStill(cursor, Storedata0, count=20)
            if clickt:
                cv2.circle(img, cursor, 15, (0, 255, 0), cv2.FILLED)
                # x0, y0, w0, h0 = 600, 400, 200, 400  # 设置初试窗口位置和大小
                # track_window = (x0, y0, w0, h0)

                x0, y0,  w0, h0= cursor[0] , cursor[1], 500, 300
                track_window = (x0-w0, y0-h0, w0-60, h0-60)

                cv2.rectangle(img, (x0 - w0, y0 - h0),(x0-60, y0-60),(200, 255, 0), 2)
                roi = img[y0-h0:y0-60, x0 - w0:x0-60]

                # roi = img[y0:y0 + h0, x0:x0 + w0]
                # roi区域的hsv图像
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                # 取值hsv值在(0,60,32)到(180,255,255)之间的部分
                mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                # 计算直方图,参数为 图片(可多)，通道数，蒙板区域，直方图长度，范围
                roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                # 归一化
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                # 设置终止条件，迭代10次或者至少移动1次
                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                clickt = False
                detecT = True
        if detecT and not clickt :
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # 计算反向投影
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # 调用meanShift算法在dst中寻找目标窗口，找到后返回目标窗口
            print('original',track_window)
            _, track_window = cv2.meanShift(dst, track_window, term_crit)
            print(track_window)
            # Draw it on image
            x0, y0, w0, h0 = track_window
            cv2.rectangle(img, (x0, y0), (x0 + w0, y0 +h0), 255, 2)


    # 显示帧率
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, 'fps: ' + str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    # modify the UI of main menu
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        x, y = rect.posCenter
        wi, hi = rect.size
        cv2.rectangle(imgNew, (x, y), (x + wi, y + hi), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (x, y, wi, hi), 20)
    cv2.putText(img, 'A', (50 + 50, 50 + 105), cv2.FONT_HERSHEY_PLAIN, 5, (100, 100, 100), 3)   # click enter entertainment mode
    cv2.putText(img, 'B', (300 + 50, 50 + 105), cv2.FONT_HERSHEY_PLAIN, 5, (100, 100, 100), 3)  # click enter office mode
    cv2.putText(img, 'C', (550 + 50, 50 + 105), cv2.FONT_HERSHEY_PLAIN, 5, (100, 100, 100), 3)
    cv2.putText(img, 'D', (800 + 50, 50 + 105), cv2.FONT_HERSHEY_PLAIN, 5, (100, 100, 100), 3)
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1-alpha, 0)[mask]  # fuse img



    # Display Image
    cv2.imshow("Image", out)
    # cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

