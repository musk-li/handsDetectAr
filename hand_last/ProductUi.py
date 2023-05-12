import cv2
from PIL import Image
import cvzone
import math
from numpy import *
from cvzone.HandTrackingModule1 import HandDetector
from cvzone.HandTrackingModule2 import HandDetector2
import numpy as np
from MethodUse import readtxt,cv2ImgAddText,fingerStill,inArea,distance
import time
import sys
from yolo import YOLO
from yolo0 import YOLO0
windoww = 2280
windowh = 900
pTime = 0

lmList = []
yolo = YOLO()
yolo0 = YOLO0()
myEquation = ''
delayCounter = 0

center = [400,300]
centermap = [700,300]
wHalf = 210
hHalf = 200
imgmap = cv2.imread('map.jpg')
img_resize = cv2.resize(imgmap, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
wHalfmap = img_resize.shape[1]/2
hHalfmap = img_resize.shape[0]/2


Storedata1 = []
Storedata2 = []
Storedata3 = []
Storedata4 = []
fourclick = False
frontInter = [15,20,25,35]
frontSize = [10,15,20,30]
FrontInter = frontInter[2]
FrontSize = frontSize[2]
text_line = [['第一师接到作战命令：进攻北城，', '第一师指派98名士兵参与作战，', '第一师准备在2号公路阻击敌人，', '完成作战任务后，在南城汇合']]

myEquation = ''
delayCounter = 0

L = 500
M = 500
N = 100
D = 300
xita = math.pi / 180 * 60  # 角度xita
P = 2
r = 3
Q = 2.5  # 三点透视参数
five = math.pi / 180 * 60  # 角度five，三点透视参数
CenterZ = [1000,300]
mulsize = 6
outl = []
Storedata31 = []
Storedata32 = []
Storedata33 = []
sizesignal = False
movesignal = False
points = [[0, 0, 200, 1], [200, 0, 200, 1], [200, 0, 0, 1], [0, 0, 0, 1],
          [0, 200, 200, 1], [200, 200, 200, 1], [200, 200, 0, 1],
          [0, 200, 0, 1]]

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
recmap = BackgroundRect(centermap,wHalfmap,hHalfmap)

cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1.set(3, windoww)
cap1.set(4, windowh)
detector = HandDetector(detectionCon=0.8, maxHands=2)

cap2 = cv2.VideoCapture(1,cv2.CAP_DSHOW)
cap2.set(3, windoww)
cap2.set(4, windowh)
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector2 = HandDetector2(detectionCon=0.8, maxHands=2)

imgmap0 = cv2.imread('iconImage/0.jpg')
img_resize0 = cv2.resize(imgmap0, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap00 = cv2.imread('iconImage/00.png')
img_resize00 = cv2.resize(imgmap00, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap1 = cv2.imread('iconImage/1.png')
img_resize1 = cv2.resize(imgmap1, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap11 = cv2.imread('iconImage/1.1.png')
img_resize11 = cv2.resize(imgmap11, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap12 = cv2.imread('iconImage/1.2.png')
img_resize12 = cv2.resize(imgmap12, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap2 = cv2.imread('iconImage/2.png')
img_resize2 = cv2.resize(imgmap2, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap21 = cv2.imread('iconImage/2.1.png')
img_resize21 = cv2.resize(imgmap21, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap22 = cv2.imread('iconImage/2.2.png')
img_resize22 = cv2.resize(imgmap22, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap23 = cv2.imread('iconImage/2.3.png')
img_resize23 = cv2.resize(imgmap23, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap3 = cv2.imread('iconImage/3.png')
img_resize3 = cv2.resize(imgmap3, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap31 = cv2.imread('iconImage/3.1.png')
img_resize31 = cv2.resize(imgmap31, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap32 = cv2.imread('iconImage/3.2.png')
img_resize32 = cv2.resize(imgmap32, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap4 = cv2.imread('iconImage/4.png')
img_resize4 = cv2.resize(imgmap4, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap41 = cv2.imread('iconImage/4.1.png')
img_resize41 = cv2.resize(imgmap41, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
mode = 'home'
iconx = 400
backIcon = True
backPosition = [100,200,400,500]
while True:
    Rate1 = 1
    success1, img = cap1.read()
    # img = cv2.flip(img, 1)
    allHands, img = detector.findHands(img, draw=True, flipType=True)
    l = 200
    if len(allHands) != 0:
        lmList = allHands[0]['lmList']
        cursor = lmList[8]
        l, _, _ = detector.findDistance(lmList[8], lmList[12], img)
        if not backIcon and not ( iconx+150< cursor[0] < iconx+250 or 550< cursor[1] < 650):
            backIcon = not backIcon
    if mode == 'home':
        if len(allHands) != 0:
            lmList = allHands[0]['lmList']
            cursor = lmList[8]
            l, _, _ = detector.findDistance(lmList[8], lmList[12], img)
            if l < 50 and iconx+200 < cursor[0] < iconx+300 and 400 < cursor[1] < 500:
                    Rate1 = 1
                    mode = 'function'
            else:
                blk = np.zeros(img.shape, np.uint8)
                blk[400:500, iconx+200:iconx + 300] = img_resize0[:, :, :]
                img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)
        else:
            blk = np.zeros(img.shape, np.uint8)
            blk[400:500, iconx+200:iconx + 300] = img_resize0[:, :, :]
            img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)
    if mode == 'function':
        if l < 50 and iconx+150 < cursor[0] < iconx + 250 and 550 < cursor[1] < 650 and backIcon:
            mode = 'home'
            backIcon = not backIcon
        elif l < 50 and iconx  < cursor[0] < iconx + 100 and 550 < cursor[1] < 650:
            mode = 'detect_pic'
        elif l < 50 and iconx < cursor[0] < iconx + 100 and 400 < cursor[1] < 500:
            mode = 'Remessage'
        elif l < 50 and iconx < cursor[0] < iconx + 100 and 250 < cursor[1] < 350:
            mode = 'ReOrder'
        elif l < 50 and iconx < cursor[0] < iconx + 100 and 100 < cursor[1] < 200:
            mode = 'RemoteTrack'
            detecT = False
            clickt = False
        else:
            blk = np.zeros(img.shape, np.uint8)
            blk[550:650, iconx+150:iconx+250] = img_resize00[:, :, :]
            blk[550:650, iconx:iconx+100] = img_resize1[:, :, :]
            blk[400:500, iconx:iconx+100] = img_resize2[:, :, :]
            blk[250:350, iconx:iconx+100] = img_resize3[:, :, :]
            blk[100:200, iconx:iconx+100] = img_resize4[:, :, :]
            img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)

    if mode == 'detect_pic':
        if l < 50 and iconx+150 < cursor[0] < iconx + 250 and 550 < cursor[1] < 650 and backIcon:
            mode = 'function'
            backIcon = not backIcon
        elif l < 50 and iconx < cursor[0] < iconx + 100 and 400 < cursor[1] < 500:
            mode = 'detect_all'
        elif l < 50 and iconx < cursor[0] < iconx + 100 and 250 < cursor[1] < 350:
            mode = 'detect_part'
        else:
            blk = np.zeros(img.shape, np.uint8)
            blk[550:650, iconx+150:iconx+250] = img_resize00[:, :, :]
            blk[400:500, iconx:iconx+100] = img_resize11[:, :, :]
            blk[250:350, iconx:iconx+100] = img_resize12[:, :, :]
            img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)

    if mode == 'Remessage':
        if l < 50 and iconx+150 < cursor[0] < iconx + 250 and 550 < cursor[1] < 650 and backIcon:
            mode = 'function'
            backIcon = not backIcon
        elif l < 50 and iconx < cursor[0] < iconx + 100 and 550 < cursor[1] < 650:
            mode = 'Remessage_read'
        elif l < 50 and iconx < cursor[0] < iconx + 100 and 300 < cursor[1] < 400:
            mode = 'Remessage_cacu'
        elif l < 50 and iconx < cursor[0] < iconx + 100 and 150 < cursor[1] < 250:
            mode = 'Remessage_3D'
        else:
            blk = np.zeros(img.shape, np.uint8)
            blk[550:650, iconx+150:iconx+250] = img_resize00[:, :, :]
            blk[550:650, iconx:iconx+100] = img_resize23[:, :, :]
            blk[300:400, iconx:iconx+100] = img_resize22[:, :, :]
            blk[150:250, iconx:iconx + 100] = img_resize21[:, :, :]
            img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)

    if mode == 'ReOrder':
        if l < 50 and iconx+150 < cursor[0] < iconx + 250 and 550 < cursor[1] < 650 and backIcon:
            mode = 'function'
            backIcon = not backIcon
        elif l < 50 and iconx < cursor[0] < iconx + 100 and 550 < cursor[1] < 650:
            mode = 'ReOrder_fixed'
        elif l < 50 and iconx < cursor[0] < iconx + 100 and 400 < cursor[1] < 500:
            mode = 'ReOrder_match'
        else:
            blk = np.zeros(img.shape, np.uint8)
            blk[550:650, iconx+150:iconx+250] = img_resize00[:, :, :]
            blk[550:650, iconx:iconx+100] = img_resize31[:, :, :]
            blk[400:500, iconx:iconx + 100] = img_resize32[:, :, :]
            img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)

    if mode == 'RemoteTrack':
        if l < 50 and iconx+150 < cursor[0] < iconx + 250 and 550 < cursor[1] < 650 and backIcon:
            mode = 'function'
            backIcon = not backIcon
        elif l < 50 and iconx < cursor[0] < iconx + 100 and 250 < cursor[1] < 350:
            mode = 'RemoteTrack_duoji'
        else:
            blk = np.zeros(img.shape, np.uint8)
            blk[550:650, iconx+150:iconx+250] = img_resize00[:, :, :]
            blk[250:350, iconx:iconx + 100] = img_resize41[:, :, :]
            img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)

    if mode == 'detect_part':
        if len(allHands) != 0:
            # lmList = allHands[0]['lmList']
            cursor = lmList[8]
            # l, _, _ = detector.findDistance(lmList[8],lmList[12],img)
            m_b, _, _ = detector.findDistance(lmList[4],lmList[8], img)
            if l < 50:
                win_size = 320
                if cursor[0] > win_size and cursor[1] > win_size:
                    box_left = cursor[0] - win_size
                    box_top = cursor[1] - win_size
                    cv2.rectangle(img, (box_left, box_top), cursor, (0, 255, 0), 2)
                    snip_img = img[box_top:cursor[1], box_left:cursor[0]]

                    if m_b < 50:
                        cv2.rectangle(img, (box_left, box_top), cursor, (255, 255, 0), 2)
                        cv2.imwrite('snip.jpg', snip_img)
                        img_temp = Image.open('snip.jpg')
                        res_image, out_class = yolo.detect_image(img_temp)
                        cv2.putText(img, str(out_class), (box_left, box_top), cv2.FONT_HERSHEY_PLAIN,
                                    2, (0, 0, 255), 2)
                else:
                    cv2.putText(img, str('out of the range'), (5, 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (0, 0, 255), 2)
            if l < 50 and backPosition[0] < cursor[0] < backPosition[1] and backPosition[2] < cursor[1] < backPosition[3] and backIcon:
                mode = 'detect_pic'
                backIcon = not backIcon
        blk = np.zeros(img.shape, np.uint8)
        blk[backPosition[2]:backPosition[3], backPosition[0]:backPosition[1]] = img_resize00[:, :, :]
        img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)
    if mode == 'detect_all':
        if len(allHands) != 0:
            # lmList = allHands[0]['lmList']
            # l, _, _ = detector.findDistance(lmList[8],lmList[12],img)
            if l < 50 and backPosition[0] < cursor[0] < backPosition[1] and backPosition[2] < cursor[1] < backPosition[3] and backIcon:
                mode = 'detect_pic'
                backIcon = not backIcon
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(np.uint8(img))
        img = np.array(yolo0.detect_image(img))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        blk = np.zeros(img.shape, np.uint8)
        blk[backPosition[2]:backPosition[3], backPosition[0]:backPosition[1]] = img_resize00[:, :, :]
        img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)
    if mode == 'Remessage_read':
        RecLeftup = (rec.center[0] - rec.w, rec.center[1] - rec.h)
        RecRightdown = (rec.center[0] + rec.w, rec.center[1] + rec.h)
        RecLeftupmap = (int(recmap.center[0] - recmap.w), int(recmap.center[1] - recmap.h))
        RecRightdownmap = (int(recmap.center[0] + recmap.w), int(recmap.center[1] + recmap.h))
        if len(allHands) != 0:
            if l < 50 and backPosition[0] < cursor[0] < backPosition[1] and backPosition[2] < cursor[1] < backPosition[3] and backIcon:
                mode = 'Remessage'
                backIcon = not backIcon

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
                rec8 = inArea(lmList[8], RecLeftup, RecRightdown)
                rec12 = inArea(lmList[12], RecLeftup, RecRightdown)
                recmap8 = inArea(lmList[8], RecLeftupmap, RecRightdownmap)
                recmap12 = inArea(lmList[12], RecLeftupmap, RecRightdownmap)

                Indesfingermap = inArea(lmList[8], RecLeftupmap, RecRightdownmap)
                if l<50 and rec8 and rec12:
                    rec.update(finger8)
                if l<50 and recmap8 and recmap12 and not (rec8 and rec12):
                    recmap.update(finger8)


        xi = RecLeftup[0]
        yi = RecLeftup[1] - 25
        blk = np.zeros(img.shape, np.uint8)
        cv2.rectangle(blk, RecLeftup, RecRightdown, (255, 255, 255), -1)
        img = cv2.addWeighted(img, 1.0, blk, 0.8, 1)

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
        # text_line = readtxt('ReadChinese.txt', 14)
        readF = text_line[0]
        for i, text in enumerate(readF):
            if text:
                img = cv2ImgAddText(img, text, xi, yi + FrontInter * (i + 1) + 15, (0, 0, 139), FrontSize)
        if inArea(RecLeftupmap, (0, 0), (1250, 900)) and inArea(RecRightdownmap, (0, 0), (1250, 900)):
            img[RecLeftupmap[1]:RecRightdownmap[1], RecLeftupmap[0]:RecRightdownmap[0]] = img_resize[:, :, :]

        else:
            recmap = BackgroundRect(centermap,wHalfmap,hHalfmap)
            cv2.putText(img, 'Out of range', (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 0, 255), 3)
        blk = np.zeros(img.shape, np.uint8)
        blk[backPosition[2]:backPosition[3], backPosition[0]:backPosition[1]] = img_resize00[:, :, :]
        img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)
    if mode == 'Remessage_cacu':
        if l < 50 and backPosition[0] < cursor[0] < backPosition[1] and backPosition[2] < cursor[1] < backPosition[3] and backIcon:
            mode = 'Remessage'
            backIcon = not backIcon
        class Button:
            def __init__(self, pos, width, height, value):
                self.pos = pos
                self.width = width
                self.height = height
                self.value = value
            def draw(self, img):
                cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                              (225, 225, 225), cv2.FILLED)
                cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                              (50, 50, 50), 3)
                cv2.putText(img, self.value, (self.pos[0] + 40, self.pos[1] + 60), cv2.FONT_HERSHEY_PLAIN,
                            2, (50, 50, 50), 2)
            def checkClick(self, x, y):
                if self.pos[0] < x < self.pos[0] + self.width and \
                        self.pos[1] < y < self.pos[1] + self.height:
                    cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                                  (255, 255, 255), cv2.FILLED)  # cv2.rectangle(img,pt1,pt2,color,thickness)
                    cv2.rectangle(img, self.pos, (self.pos[0] + self.width, self.pos[1] + self.height),
                                  (50, 50, 50), 3)

                    cv2.putText(img, self.value, (self.pos[0] + 25, self.pos[1] + 80), cv2.FONT_HERSHEY_PLAIN,
                                5, (0, 0, 0), 5)
                    return True
                else:
                    return False
        buttonListValues = [['7', '8', '9', '*'],
                            ['4', '5', '6', '-'],
                            ['1', '2', '3', '+'],
                            ['0', '/', '.', '=']]
        buttonList = []
        for x in range(4):
            for y in range(4):
                xpos = x * 100 + 800
                ypos = y * 100 + 150
                buttonList.append(Button((xpos, ypos), 100, 100, buttonListValues[y][x]))

        # Draw all buttons
        cv2.rectangle(img, (800, 50), (800 + 400, 70 + 100),
                      (225, 225, 225), cv2.FILLED)
        cv2.rectangle(img, (800, 50), (800 + 400, 70 + 100),
                      (50, 50, 50), 3)
        for button in buttonList:
            button.draw(img)
        if allHands:

            lmList = allHands[0]['lmList']
            length, _, img = detector.findDistance(lmList[8], lmList[12], img)
            x, y = lmList[8]
            if length < 50 and delayCounter == 0:
                for i, button in enumerate(buttonList):
                    if button.checkClick(x, y):
                        myValue = buttonListValues[int(i % 4)][int(i / 4)]
                        if myValue == "=":
                            myEquation = str(eval(myEquation))
                        else:
                            myEquation += myValue
                        delayCounter = 1
        if delayCounter != 0:
            delayCounter += 1
            if delayCounter > 10:
                delayCounter = 0
        cv2.putText(img, myEquation, (810, 120), cv2.FONT_HERSHEY_PLAIN,
                    3, (50, 50, 50), 3)
        blk = np.zeros(img.shape, np.uint8)
        blk[backPosition[2]:backPosition[3], backPosition[0]:backPosition[1]] = img_resize00[:, :, :]
        img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)
    if mode == 'Remessage_3D':
        out0 = []
        Sumpoint = 0
        if len(allHands) != 0:
            if l < 50 and backPosition[0] < cursor[0] < backPosition[1] and backPosition[2] < cursor[1] < backPosition[3] and backIcon:
                mode = 'Remessage'
                backIcon = not backIcon
            lmList = allHands[0]['lmList']
            Are = 150
            RU = [CenterZ[0] - Are, CenterZ[1] - Are]
            RD = [CenterZ[0] + Are, CenterZ[1] + Are]
            if len(allHands) == 2:
                if allHands[0]['type'] == 'Left':
                    lmListLeft = allHands[0]['lmList']
                    lmListRight = allHands[1]['lmList']
                else:
                    lmListLeft = allHands[1]['lmList']
                    lmListRight = allHands[0]['lmList']
                Lindexfin1 = lmListLeft[8]
                Rindexfin1 = lmListRight[8]
                Lindexfinger1 = inArea(lmListLeft[8], RU, RD)
                Rindexfinger1 = inArea(lmListRight[8], RU, RD)
                Storedata1, clickLt1 = fingerStill(Lindexfin1, Storedata1, count=18)
                Storedata2, clickLi1 = fingerStill(Rindexfin1, Storedata2, count=18)
                if clickLt1 and clickLi1:
                    sizesignal = not sizesignal
                if sizesignal:
                    cv2.circle(img, (500, 50), 15, (0, 255, 0), cv2.FILLED)
                    m, _, _ = detector.findDistance(Lindexfin1, Rindexfin1, img)
                    mulsize = int(m / 100)
                if not sizesignal:
                    cv2.circle(img, (500, 50), 15, (0, 0, 255), cv2.FILLED)
            else:
                lmListRight = allHands[0]['lmList']
                # cv2.rectangle(img, RU, RD, (255, 0, 0), 2, 1)
                Storedata3, click2 = fingerStill(lmListRight[8], Storedata3, count=15)
                Rindexfinger2 = inArea(lmListRight[8], RU, RD)
                if click2 and Rindexfinger2:
                    movesignal = not movesignal
                if movesignal:
                    CenterZ = lmListRight[8]
                    cv2.circle(img, (500, 50), 15, (0, 255, 0), cv2.FILLED)

            l, _, _ = detector.findDistance(lmListRight[4], lmListRight[8], img)
            theta = l * 360 / 280
            cosr = np.cos(theta * np.pi / 180)
            sinr = np.sin(theta * np.pi / 180)
            Rx = np.array([[1, 0, 0, 0], [0, cosr, -sinr, 0], [0, sinr, cosr, 0], [0, 0, 0, 1]])
            Ry = np.array([[cosr, 0, sinr, 0], [0, 1, 0, 0], [-sinr, 0, cosr, 0], [0, 0, 0, 1]])
            Rz = np.array([[cosr, -sinr, 0, 800], [sinr, cosr, 0, 300], [0, 0, 1, 0], [0, 0, 0, 1]])
            mtranslation = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [L, M, N, 1]], dtype='float32')
            # 沿y轴旋转
            mrotatey = np.array([[math.cos(xita), 0, -math.sin(xita), 0], [0, 1, 0, 0],
                                 [math.sin(xita), 0, math.cos(xita), 0], [0, 0, 0, 1]],
                                dtype='float32')
            # 沿x轴旋转
            mrotatex = np.array(
                [[1, 0, 0, 0], [0, math.cos(five), math.sin(five), 0],
                 [0, -math.sin(five), math.cos(five), 0], [0, 0, 0, 1]],
                dtype='float32')
            # 透视
            mperspective = np.array([[1, 0, 0, P], [0, 1, 0, Q], [0, 0, 1, r], [0, 0, 0, 1]], dtype='float32')
            # 向xoy平面做正投影
            mprojectionxy = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype='float32')
            threeMatrix = np.matmul(
                np.matmul(np.matmul(np.matmul(mtranslation, mrotatey), mrotatex),
                          mperspective), mprojectionxy)
            for e in range(0, len(points)):
                mitrip0 = np.dot(points[e], Rx.transpose())
                mitrip1 = np.dot(mitrip0, Ry.transpose())
                mitrip2 = np.dot(mitrip1, Rz.transpose())
                tmp = np.matmul(np.array(mitrip2, dtype='float32'), threeMatrix)
                tmp = tmp / tmp[3]
                tmp = tmp * 1000
                poinsition = np.round(tmp[:2]).astype(int)
                Sumpoint = Sumpoint + poinsition
                centerPoint = np.round(Sumpoint / 8).astype(int)
                out0.append(poinsition)
            newZero = out0 - centerPoint
            mulnewzero = newZero * mulsize
            outl = mulnewzero + np.array(CenterZ)
        if len(outl):
            cv2.line(img, outl[0], outl[1], (0, 255, 0), 3)
            cv2.line(img, outl[0], outl[3], (0, 255, 0), 3)
            cv2.line(img, outl[1], outl[2], (0, 255, 0), 3)
            cv2.line(img, outl[2], outl[3], (0, 255, 0), 3)

            cv2.line(img, outl[4], outl[5], (0, 255, 0), 3)
            cv2.line(img, outl[5], outl[6], (0, 255, 0), 3)
            cv2.line(img, outl[4], outl[7], (0, 255, 0), 3)
            cv2.line(img, outl[6], outl[7], (0, 255, 0), 3)

            cv2.line(img, outl[0], outl[4], (0, 255, 0), 3)
            cv2.line(img, outl[1], outl[5], (0, 255, 0), 3)
            cv2.line(img, outl[2], outl[6], (0, 255, 0), 3)
            cv2.line(img, outl[3], outl[7], (0, 255, 0), 3)
            pointsl = np.array([tuple(outl[0]), tuple(outl[1]), tuple(outl[2]), tuple(outl[3])])
            points2 = np.array([tuple(outl[4]), tuple(outl[5]), tuple(outl[6]), tuple(outl[7])])
            img = cv2.fillPoly(img, [pointsl], [100, 0, 250])
            img = cv2.fillPoly(img, [points2], [1400, 200, 250])
            blk = np.zeros(img.shape, np.uint8)
            blk[backPosition[2]:backPosition[3], backPosition[0]:backPosition[1]] = img_resize00[:, :, :]
            img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)
    # if mode == 'ReOrder_fixed':
    # if mode == 'ReOrder_match':
    if mode == 'RemoteTrack_duoji':
        suc2, img2 = cap2.read()
        m_b, _, _ = detector.findDistance(lmList[4], lmList[8], img)
        img1 = cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        img2 = cv2.resize(img2, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
        allHands, img1, img2 = detector2.findHands(img1, img2, draw=True, flipType=True)
        if len(allHands) != 0:
            cv2.circle(img2, allHands[0]['lmList'][8], 15, (255, 0, 255), cv2.FILLED)
            lmList = allHands[0]['lmList']
            cursor = lmList[8]
            l, _, _ = detector.findDistance(lmList[8], lmList[12], img)

            if l < 50 and backPosition[0] < cursor[0] < backPosition[1] and backPosition[2] < cursor[1] < backPosition[3] and backIcon:
                mode = 'RemoteTrack'
                backIcon = not backIcon
            if l < 50:
                x0, y0, w0, h0 = cursor[0], cursor[1], 300, 300
                cv2.rectangle(img2, (x0 - w0, y0 - h0), (x0 - 60, y0 - 60), (200, 255, 0), 2)
                if m_b < 50:
                    clickt = True
            if clickt and cursor[0]>200:
                cv2.circle(img2, cursor, 15, (0, 255, 0), cv2.FILLED)
                x0, y0, w0, h0 = cursor[0], cursor[1], 300, 300
                track_window = (x0 - w0, y0 - h0, w0 - 60, h0 - 60)

                roi = img2[y0 - h0:y0 - 60, x0 - w0:x0 - 60]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
                term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                clickt = False
                detecT = True

        if detecT and not clickt:
            hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            _, track_window = cv2.meanShift(dst, track_window, term_crit)

            x0, y0, w0, h0 = track_window
            cv2.rectangle(img2, (x0, y0), (x0 + w0, y0 + h0), 255, 2)
        blk = np.zeros(img2.shape, np.uint8)
        blk[backPosition[2]:backPosition[3], backPosition[0]:backPosition[1]] = img_resize00[:, :, :]
        img2 = cv2.addWeighted(img2, 1.0, blk, Rate1, 1)
        img = img2

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, 'fps: ' + str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.imshow('imgshow',img)
    cv2.waitKey(1)

