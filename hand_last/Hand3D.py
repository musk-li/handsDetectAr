import cv2
import numpy as np
from cvzone.HandTrackingModule1 import HandDetector
import math
from MethodUse import readtxt,cv2ImgAddText,fingerStill,inArea,distance
# cx =  800
# cy = 400
# f = 200
# a = b = 0.4
# listpo =np.array([[0,300,300],[400,0,300],[400,600,300],[800,300,300],[0,300,800],[400,0,800],[400,600,800],[800,300,800]])
L = 500
M = 500
N = 100
D = 300  # 视距
xita = math.pi / 180 * 60  # 角度xita
P = 2
r = 3
Q = 2.5  # 三点透视参数
five = math.pi / 180 * 60  # 角度five，三点透视参数
CenterZ = [1000,300]
mulsize = 6
outl = []
Storedata1 = []
Storedata2 = []
Storedata3 = []
sizesignal = False
movesignal = False
points = [[0, 0, 200, 1], [200, 0, 200, 1], [200, 0, 0, 1], [0, 0, 0, 1],
          [0, 200, 200, 1], [200, 200, 200, 1], [200, 200, 0, 1],
          [0, 200, 0, 1]]
cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1.set(3, 2280)      # width 1280
cap1.set(4, 900)
detector = HandDetector(detectionCon=0.8, maxHands=2)
while True:
    success1, img = cap1.read()
    # img = cv2.flip(img, 1)
    allHands, img = detector.findHands(img, draw=True, flipType=True)
    out0 = []
    Sumpoint = 0
    if len(allHands) != 0:
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
            if  clickLt1 and clickLi1 :
                sizesignal = not sizesignal
            if sizesignal:
                cv2.circle(img, (500, 50), 15, (0, 255, 0), cv2.FILLED)
                m, _, _ = detector.findDistance(Lindexfin1, Rindexfin1, img)
                mulsize = int(m/100)
            if not sizesignal:
                cv2.circle(img, (500, 50), 15, (0, 0,255), cv2.FILLED)
        else:
            lmListRight = allHands[0]['lmList']
            cv2.rectangle(img, RU, RD, (255, 0, 0), 2, 1)
            Storedata3, click2 = fingerStill(lmListRight[8], Storedata3, count=15)
            Rindexfinger2 = inArea(lmListRight[8], RU, RD)
            if click2 and Rindexfinger2:
                movesignal = not movesignal
            if movesignal:
                CenterZ = lmListRight[8]
                cv2.circle(img, (500, 50), 15, (0, 255, 0), cv2.FILLED)

        l, _, _ = detector.findDistance(lmListRight[4], lmListRight[8], img)
        theta = l*360/280
        cosr = np.cos(theta * np.pi / 180)
        sinr = np.sin(theta * np.pi / 180)

        # R = np.array([[cosr, -sinr, 0],
        #               [sinr, cosr, 0], [0, 0, 1]])
        # R = np.array([[1,1-cosr-sinr,1-cosr+sinr,400],[1-cosr+sinr,1,1-sinr-cosr,300],[1-cosr-sinr,1-cosr+sinr,1,0],[0,0,0,1]])
        # R = np.array([[cosr,-sinr,0,100],[sinr,cosr,0,100],[0,0,1,0],[0,0,0,1]])
        '''x轴'''
        Rx = np.array([[1,0, 0, 0], [0, cosr, -sinr, 0], [0, sinr, cosr, 0], [0, 0, 0, 1]])
        '''y轴'''
        Ry = np.array([[cosr, 0,sinr, 0], [0,1,0, 0], [-sinr, 0, cosr, 0], [0, 0, 0, 1]])
        '''z轴'''
        Rz = np.array([[cosr, -sinr, 0, 800], [sinr, cosr, 0, 300], [0, 0, 1, 0], [0, 0, 0, 1]])
        #一点透视
        # transMatrix = np.array(
        #     [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1 / D], [L, M, 0, 1 + (N / D)]],
        #     dtype='float32')
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

        # 一点透视：将三维坐标映射到二维
        for e in range(0, len(points)):
            # mitrip = np.dot(points[e], R.transpose())
            # tmp = np.matmul(np.array(mitrip, dtype='float32'), transMatrix)
            # tmp = tmp / tmp[3]  # 齐次化
            # outl.append(np.round(tmp[:2]).astype(int))

            mitrip0 = np.dot(points[e], Rx.transpose())
            mitrip1 = np.dot(mitrip0, Ry.transpose())
            mitrip2 = np.dot(mitrip1, Rz.transpose())
            tmp = np.matmul(np.array(mitrip2, dtype='float32'), threeMatrix)
            tmp = tmp / tmp[3]  # 齐次化
            tmp = tmp * 1000  # 非必须：放大，太小看不见
            poinsition = np.round(tmp[:2]).astype(int)
            Sumpoint = Sumpoint + poinsition
            centerPoint = np.round(Sumpoint/8).astype(int)
            out0.append(poinsition)
        newZero = out0 - centerPoint
        mulnewzero = newZero*mulsize
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
        # for i in range(len(listpo)):
        #     x, y, z = np.dot(listpo[i], R.transpose()) + np.array([0, 0, 0])
        #     K = np.array([[a * f, 0, cx], [0, a * f, cy], [400, 300, 1]])
        #     p = np.array([x, y, z])
        #     out = np.dot(K, p) / z
        #     out = np.round(out).astype(int)
        #     drawout = out[:2]
        #     outl.append(drawout)
        #     cv2.circle(img, drawout, 7, (0, 255, 0), cv2.FILLED)
        # cv2.line(img, outl[0], outl[1], (0, 255, 0), 3)
        # cv2.line(img, outl[0], outl[2], (0, 255, 0), 3)
        # cv2.line(img, outl[1], outl[3], (0, 255, 0), 3)
        # cv2.line(img, outl[2], outl[3], (0, 255, 0), 3)
        #
        # cv2.line(img, outl[4], outl[5], (0, 255, 0), 3)
        # cv2.line(img, outl[4], outl[6], (0, 255, 0), 3)
        # cv2.line(img, outl[5], outl[7], (0, 255, 0), 3)
        # cv2.line(img, outl[6], outl[7], (0, 255, 0), 3)
        #
        # cv2.line(img, outl[0], outl[4], (0, 255, 0), 3)
        # cv2.line(img, outl[1], outl[5], (0, 255, 0), 3)
        # cv2.line(img, outl[2], outl[6], (0, 255, 0), 3)
        # cv2.line(img, outl[3], outl[7], (0, 255, 0), 3)
        pointsl = np.array([tuple(outl[0]),tuple(outl[1]),tuple(outl[2]),tuple(outl[3])])
        points2 = np.array([tuple(outl[4]), tuple(outl[5]), tuple(outl[6]), tuple(outl[7])])
        img = cv2.fillPoly(img, [pointsl], [100,0,250])
        img = cv2.fillPoly(img, [points2], [1400, 200, 250])
    cv2.imshow('img',img)
    cv2.waitKey(1)




