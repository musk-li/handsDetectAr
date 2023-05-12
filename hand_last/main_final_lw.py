# coding=utf-8
import cv2
from PIL import Image
import cvzone
from numpy import *
from cvzone.HandTrackingModule import HandDetector
# from HandTrackingModule1 import HandDetector
import numpy as np
from MethodUse import readtxt, cv2ImgAddText, fingerStill, inArea, distance
import time
from yolo import YOLO
import sys

# Set parameters
lmList = []
w, h = 100, 100             # the size of main menu
colorR = (0, 0, 255)        # the default color of main menu
mode = ''                   # initialize mode

yolo = YOLO()               # initialize detection

# Variables of Keyboard
myEquation = ''
delayCounter = 0

# Webcam
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3, 1280)            # width 1280
cap.set(4, 720)             # height 720
pTime = 0
detector = HandDetector(detectionCon=0.8, maxHands=2)  # confidence and the num of hands

# Parameters of text scaling
center = [410, 400]
wHalf = 310
hHalf = 300
Storedata = []
Storedata0 = []
Storedata1 = []
Storedata2 = []
Storedata3 = []
Storedata4 = []
clickopen = False
fourclick = False
frontInter = [15, 20, 25, 35]
frontSize = [10, 15, 20, 30]
FrontInter = frontInter[2]
FrontSize = frontSize[2]

# Parameters of target tracking
okkk = 0

# define rectangle background of text scaling
class BackgroundRect():
    def __init__(self, center, w, h):
        self.center = center
        self.w = w
        self.h = h

    def update(self, cursor):
        cx, cy = self.center
        w = self.w
        h = self.h
        if cx - w < cursor[0] < cx + w and \
                cy - h < cursor[1] < cy + h:
            self.center = cursor

    def updateSize(self, whalf, hhalf):
        self.w = whalf
        self.h = hhalf


rec = BackgroundRect(center, wHalf, hHalf)

# augmented reality setting
imgTarget = cv2.imread('TargetImage.jpg')
myVid = cv2.VideoCapture('video.mp4')

detection = False
frameCounter = 0

success, imgVideo = myVid.read()
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

# Initiate ORB detector
orb = cv2.ORB_create(nfeatures=1000)
# find the query image keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(imgTarget, None)
# imgTarget = cv2.drawKeypoints(imgTarget, kp1, None)


def stackImages(imgArray, scale, lables=[]):
    sizeW = imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((sizeH, sizeW, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, lables[d], (eachImgWidth * c + 10, eachImgHeight * d + 20), cv2.FONT_HERSHEY_COMPLEX,
                            0.7, (255, 0, 255), 2)
    return ver


# Draw main menu rectangle
class DeagRect():
    def __init__(self, posCenter, size=[w, h]):
        self.posCenter = posCenter
        self.size = size


rectList = []
for x in range(5):                                  # to generate four rectangles
    rectList.append(DeagRect([x * 200 + 50, 50]))   # left_top (50,50) (250,50) (450,50) (650,50) (850,50)

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
            if 50 < cursor[0] < 50 + w and 50 < cursor[1] < 50 + h:
                colorR = [255, 0, 0]
                mode = 'detect_pic'     # Entertainment: Target recognition [blue]
            elif 250 < cursor[0] < 250 + w and 50 < cursor[1] < 50 + h:
                colorR = [0, 255, 255]
                mode = 'calculate'      # Office: Calculator [yellow]
            elif 450 < cursor[0] < 450 + w and 50 < cursor[1] < 50 + h:
                colorR = [226, 43, 138]
                mode = 'bigsmall'       # [purple]
            elif 650 < cursor[0] < 650 + w and 50 < cursor[1] < 50 + h:
                colorR = [0, 255, 0]
                mode = 'trackobject'    # [green]
                if okkk == 1:
                    del tracker
                    okkk = 0
                else:
                    pass
            elif 850 < cursor[0] < 850 + w and 50 < cursor[1] < 50 + h:
                colorR = [255, 255, 0]
                mode = 'ar'             # [cyan]

    # Entertainment: Target recognition
    if mode == "detect_pic":
        # # Get image from webcam (frame)
        # success, img = cap.read()
        # img = cv2.flip(img, 1)  # 1 flip horizontally

        # # Detection of hand
        # allHands, img = detector.findHands(img,flipType=False)

        # Check for Hand
        if len(allHands) != 0:
            lmList = allHands[0]['lmList']
            cursor = lmList[8]                                              # the location of index finger
            l, _, _ = detector.findDistance(lmList[8], lmList[12], img)     # return the distance of index and middle finger
            m_b, _, _ = detector.findDistance(lmList[4], lmList[8], img)    # return the distance of index finger and thumb
            # trigger detection box
            if l < 50:
                win_size = 416  # the size of detection box
                if cursor[0] > win_size and cursor[1] > win_size:  # border condition
                    box_left = cursor[0] - win_size
                    box_top = cursor[1] - win_size
                    cv2.rectangle(img, (box_left, box_top), cursor, (0, 255, 0),
                                  2)  # cv2.rectangle(img,pt1,pt2,color,thickness) 边框
                    snip_img = img[box_top:cursor[1], box_left:cursor[0]]  # capture pic (cv2 BGR)
                    # start detection
                    if m_b < 50:  # and delayCounter == 0:
                        cv2.rectangle(img, (box_left, box_top), cursor, (255, 255, 0),
                                      2)  # cv2.rectangle(img,pt1,pt2,color,thickness) 边框
                        cv2.imwrite('snip.jpg', snip_img)
                        img_temp = Image.open('snip.jpg')
                        res_image, out_class = yolo.detect_image(img_temp)
                        # res_image.show()
                        cv2.putText(img, str(out_class), (box_left, box_top), cv2.FONT_HERSHEY_PLAIN,
                                    2, (0, 0, 255), 2)  # 照片/添加的文字/左下角坐标/字体/字体大小/颜色/字体粗细
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

    # Office: Calculator
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
                              (50, 50, 50), 3)  # cv2.rectangle(img,pt1,pt2,color,thickness) 边框

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
                      (50, 50, 50), 3)  # cv2.rectangle(img,pt1,pt2,color,thickness) edge

        for button in buttonList:
            button.draw(img)

        # Check for Hand
        if allHands:
            # Find distance between fingers
            lmList = allHands[0]['lmList']
            length, _, img = detector.findDistance(lmList[8], lmList[12], img)
            # print(length)
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
        if key == ord('c'):  # clear
            myEquation = ''
        elif key == ord('q'):  # quit
            break
        pass

    # Text scaling
    if mode == "bigsmall":
        # show the board of coordinate of lefttop and rightdown point
        RecLeftup = (rec.center[0] - rec.w, rec.center[1] - rec.h)
        RecRightdown = (rec.center[0] + rec.w, rec.center[1] + rec.h)

        if len(allHands) != 0:
            # if detect two hands, return left and right hand coords, scale the board
            if len(allHands) == 2:
                if allHands[0]['type'] == 'Left':
                    lmListLeft = allHands[0]['lmList']
                    lmListRight = allHands[1]['lmList']
                else:
                    lmListLeft = allHands[1]['lmList']
                    lmListRight = allHands[0]['lmList']
                # l, _, _ = detector.findDistance(lmListLeft[8], lmListRight[8], img)
                # b, _, _ = detector.findDistance(lmListLeft[4], lmListRight[4], img)
                # store four points coordinates
                Lthumbfin = lmListLeft[4]
                Lindexfin = lmListLeft[8]
                Rthumbfin = lmListRight[4]
                Rindexfin = lmListRight[8]
                # judge whether these four points in board
                Lthumbfinger = inArea(lmListLeft[4], RecLeftup, RecRightdown)
                Lindexfinger = inArea(lmListLeft[8], RecLeftup, RecRightdown)
                Rthumbfinger = inArea(lmListRight[4], RecLeftup, RecRightdown)
                Rindexfinger = inArea(lmListRight[8], RecLeftup, RecRightdown)
                # if four points static for a while, then trigger bigsamll function
                Storedata1, clickLt = fingerStill(Lthumbfin, Storedata1, count=15)
                Storedata2, clickLi = fingerStill(Lindexfin, Storedata2, count=15)
                Storedata3, clickRt = fingerStill(Rthumbfin, Storedata3, count=15)
                Storedata4, clickRi = fingerStill(Rindexfin, Storedata4, count=15)

                if clickLt and clickLi and clickRt and clickRi:
                    fourclick = not fourclick

                if Lthumbfinger and Rthumbfinger and Lindexfinger and Rindexfinger and not fourclick:
                    # update width by calculating the distance of left and right index finger
                    whalf = int(distance(Lindexfin, Rindexfin))
                    # update height by calculating the distance of right index and thumb finger
                    hhalf = int(distance(Rthumbfin, Rindexfin))
                    # print(whalf, hhalf)
                    # set four kinds of font-size
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

            # if detect one hand, return left and right hand coords, shift the board
            else:
                lmList = allHands[0]['lmList']
                finger8 = lmList[8]     # index finger
                Indesfinger = inArea(lmList[8], RecLeftup, RecRightdown)    # only judge one finger
                # Middlefinger = inArea(lmList[12], RecLeftup, RecRightdown)
                # Ringfinger = inArea(lmList[16], RecLeftup, RecRightdown)
                if inArea(lmList[8], RecLeftup, RecRightdown):
                    Storedata, click = fingerStill(finger8, Storedata, count=10)
                    if click: clickopen = not clickopen
                # indicate state of shifting by using red and green points
                if Indesfinger and clickopen:#and Middlefinger and Ringfinger
                    rec.update(finger8)
                    cv2.circle(img, (30, 50), 15, (0, 255, 0), cv2.FILLED)
                if not clickopen:
                    cv2.circle(img, (30, 50), 15, (0, 0, 255), cv2.FILLED)

        # set transparent of board
        xi = RecLeftup[0]
        yi = RecLeftup[1] - 25
        blk = np.zeros(img.shape, np.uint8)
        cv2.rectangle(blk, RecLeftup, RecRightdown, (255, 255, 255), -1)
        img = cv2.addWeighted(img, 1.0, blk, 0.5, 1)

        # if four points are in board, show four points by red or green points
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
        # read txt
        text_line = readtxt('ReadChinese.txt', 30)
        readF = text_line[0] \
            # +text_line[1]+text_line[2]+text_line[3]

        # show the txt row by row
        for i, text in enumerate(readF):
            if text:
                img = cv2ImgAddText(img, text, xi, yi + FrontInter * (i + 1) + 15, (0, 0, 139), FrontSize)

    # Target tracking
    if mode == "trackobject":
        # cv2.putText(img, 'trackobject', (700 + 50, 50 + 105), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 3)
        if len(allHands) != 0:
            lmList = allHands[0]['lmList']
            cursor = lmList[8]
            # if one hand static for a while, trigger track function
            Storedata0, click = fingerStill(cursor, Storedata0, count=20)
            if click:
                cv2.circle(img, cursor, 15, (0, 255, 0), cv2.FILLED)
                bbox = (cursor[0] - 300, cursor[1] - 300, 300, 300)
                tracker = cv2.TrackerMIL_create()
                okk = tracker.init(img, bbox)
                okkk = 1
        if okkk:
            ok, bbox = tracker.update(img)
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(img, p1, p2, (0, 140, 255), 2, 1)
            else:
                cv2.putText(img, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Augmented reality
    # if mode == "ar":
    #     success, imgWebcam = cap.read()
    #     imgAug = imgWebcam.copy()
    #     # find the train image keypoints and descriptors with ORB
    #     kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    #     # imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)
    #
    #     if detection == False:
    #         myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)                       # read certain fps, 0 means from scratch
    #         frameCounter = 0
    #     else:
    #         if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):     # total fps = framecounter, from scratch
    #             myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #             frameCounter = 0
    #         success, imgVideo = myVid.read()
    #         imgVideo = cv2.resize(imgVideo, (wT, hT))
    #
    #     # create BFmatcher object. Brute-Force
    #     bf = cv2.BFMatcher()
    #     # match descriptors, return k matches
    #     matches = bf.knnMatch(des1, des2, k=2)
    #     # apply ratio test
    #     good = []
    #     for m, n in matches:
    #         if m.distance < 0.75 * n.distance:
    #             good.append(m)
    #     # print(len(good))
    #     imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)
    #
    #     if len(good) > 20:
    #         detection = True
    #         srcPts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)   # kp1[m.queryIdx].pt -> coords
    #         dstPts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    #         matrix, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5)            # return Homography matrix and matched kpts, apply RANSAC to match
    #         # print(matrix)
    #
    #         pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
    #         dst = cv2.perspectiveTransform(pts, matrix)
    #         img2 = cv2.polylines(imgWebcam, [np.int32(dst)], True, (255, 0, 255), 3)    # draw polygons
    #         # direction: imgVideo to imgWebcam
    #         imgWarp = cv2.warpPerspective(imgVideo, matrix, (imgWebcam.shape[1], imgWebcam.shape[0]))
    #
    #         maskNew = np.zeros((imgWebcam.shape[0], imgWebcam.shape[1]), np.uint8)
    #         cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
    #         maskInv = cv2.bitwise_not(maskNew)
    #         imgAug = cv2.bitwise_and(imgAug, imgAug, mask=maskInv)
    #         imgAug = cv2.bitwise_or(imgWarp, imgAug)
    #         img = imgAug
    #
    #         # imgStacked = stackImages(([imgWebcam, imgVideo, imgTarget], [imgFeatures, imgWarp, imgAug]), 0.5)
    #
    #     # cv2.imshow('maskNew', imgAug)
    #     # cv2.imshow('imgWarp', imgWarp)
    #     # cv2.imshow('img2', img2)
    #     # cv2.imshow('ImgFeatures', imgFeatures)
    #     # cv2.imshow('ImgTarget', imgTarget)
    #     # cv2.imshow('myVid', imgVideo)
    #     # cv2.imshow('Webcam', imgWebcam)
    #     # cv2.imshow('imgStacked', imgStacked)
    #
    #     key = cv2.waitKey(1)
    #     frameCounter += 1
    #
    #     if key == ord('q'):
    #         break

    # display frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, 'fps: ' + str(int(fps)), (15, 30), cv2.FONT_HERSHEY_PLAIN,
                2, (255, 0, 0), 3)

    # modify the UI of main menu
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        x, y = rect.posCenter
        wi, hi = rect.size
        cv2.rectangle(imgNew, (x, y), (x + wi, y + hi), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (x, y, wi, hi), 20)
    cv2.putText(img, 'A', (50 + 25, 50 + 80), cv2.FONT_HERSHEY_PLAIN, 5, (100, 100, 100), 3)   # click enter entertainment mode
    cv2.putText(img, 'B', (250 + 25, 50 + 80), cv2.FONT_HERSHEY_PLAIN, 5, (100, 100, 100), 3)  # click enter office mode
    cv2.putText(img, 'C', (450 + 25, 50 + 80), cv2.FONT_HERSHEY_PLAIN, 5, (100, 100, 100), 3)
    cv2.putText(img, 'D', (650 + 25, 50 + 80), cv2.FONT_HERSHEY_PLAIN, 5, (100, 100, 100), 3)
    cv2.putText(img, 'E', (850 + 25, 50 + 80), cv2.FONT_HERSHEY_PLAIN, 5, (100, 100, 100), 3)
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]  # fuse img

    # Display Image
    cv2.imshow("Image", out)
    # cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
