import cv2
import time
import numpy as np
from cvzone.HandTrackingModule1 import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# initialize
detector = HandDetector(detectionCon=0.8)
startDist = None                        # 初始放缩参考距离
scale = 0                               # 初始放缩比例
cx, cy = 600, 300                       # 初始图片显示位置
img1 = cv2.imread('ImagesJPG/1.jpg')    # 初始化显示图片
newW, newH, _ = img1.shape              # 初始化显示图片 wide, height
pTime = 0                               # 初始化计时器

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)  # flipType=False (if flip(img, 1), filpType=False)
    img1 = cv2.imread('ImagesJPG/1.jpg')

    # fun1: move the pic -- 单手模式
    # if len(hands) == 1:
    #     lmList = hands[0]['lmList']
    #     # Check if clicked
    #     length, info, img = detector.findDistance(lmList[8], lmList[12], img)
    #     # print(length)
    #     if length < 60:
    #         cursor = lmList[8]
    #         # Check if in region
    #         if cx - newW // 2 < cursor[0] < cx + newW // 2 and cy - newH // 2 < cursor[1] < cy + newH // 2:
    #             cx, cy = cursor[0], cursor[1]

    # func2: scale the pic -- 双手模式
    if len(hands) == 2:
        # print(detector.fingersUp(hands[0]), detector.fingersUp(hands[1]))
        if detector.fingersUp(hands[0]) == [0, 1, 0, 0, 0] and \
                detector.fingersUp(hands[1]) == [0, 1, 0, 0, 0]:  # [1, 1, 0, 0, 0]
            lmList1 = hands[0]['lmList']
            lmList2 = hands[1]['lmList']
            # point 8 is the tip of the index finger
            if startDist is None:
                length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
                startDist = length                                                  # set the initial distance

            length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # set the compared distance
            scale = int((length - startDist) // 2)
            cx, cy = info[4:]
            print(scale)

            # sub func3: 显示放缩比例
            volBar = np.interp(scale, [-220, 220], [400, 150])   # 构建映射 [-220, 220]表示scale大致范围 不同图片需要更改 大概是图片size大小
            volPer = np.interp(scale, [-220, 220], [0, 100])     # [400, 150] [0, 100]表示映射成区间范围

            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)

    else:
        startDist = None    # if don't detect two hands, update scale strategy

    h1, w1, _ = img1.shape  # initial img size
    newH, newW = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2  # update img size

    if (cy - newH // 2) > 0 and (cy + newH // 2) < 720 and (h1 + scale) >= 2 \
            and (cx - newW // 2) > 0 and (cx + newW // 2) < 1280:                   # 边界条件，约束上下左右及缩小范围
        img1 = cv2.resize(img1, (newW, newH))
        img[cy - newH // 2:cy + newH // 2, cx - newW // 2:cx + newW // 2] = img1
    else:
        cv2.putText(img, str('out of the range'), (5, 30), cv2.FONT_HERSHEY_PLAIN,  # 超边界提示
                    2, (0, 0, 255), 2)

    # func4: calculate fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
