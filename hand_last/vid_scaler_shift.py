import cv2
import time
import numpy as np
from cvzone.HandTrackingModule1 import HandDetector

# load video
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
myVid = cv2.VideoCapture('video.mp4')

# initialize
detector = HandDetector(detectionCon=0.8, maxHands=2)
cx, cy = 1050, 230              # 初始视频中心位置坐标
win_size = 400                  # 初始视频窗大小
startDist = None                # 初始放缩参考距离
scale = 0                       # 初始放缩比例
frameCounter = 0                # 循环播放视频帧
pTime = 0                       # 初始化计时器

while True:
    # load video frame1
    ret1, imgWebcam = cap.read()
    imgWebcam = cv2.flip(imgWebcam, 1)  # 转变视角时删除

    # 循环播放视频
    if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):  # 获取总帧数，如果计数帧=总帧数，设置0帧重新播放视频
        myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0

    # load video frame2
    ret2, imgVideo = myVid.read()

    # hand detection
    allHands, imgWebcam = detector.findHands(imgWebcam, flipType=False)  # 转变视角时更改

    # fun1: move the vid -- 单手模式
    if len(allHands) == 1:
        lmList = allHands[0]['lmList']
        # Check if clicked
        length, info, imgWebcam = detector.findDistance(lmList[8], lmList[12], imgWebcam)
        # print(length)
        if length < 60:
            cursor = lmList[8]  # the location of index finger
            # Check if in region
            if cx - win_size // 2 < cursor[0] < cx + win_size // 2 and cy - win_size // 2 < cursor[
                1] < cy + win_size // 2:
                cx, cy = cursor[0], cursor[1]

    # func2: scale the vid -- 双手模式
    if len(allHands) == 2:
        # print(detector.fingersUp(hands[0]), detector.fingersUp(hands[1]))
        if detector.fingersUp(allHands[0]) == [0, 1, 0, 0, 0] and \
                detector.fingersUp(allHands[1]) == [0, 1, 0, 0, 0]:  # [1, 1, 0, 0, 0]
            lmList1 = allHands[0]['lmList']
            lmList2 = allHands[1]['lmList']
            # point 8 is the tip of the index finger
            if startDist is None:
                length, info, img = detector.findDistance(lmList1[8], lmList2[8], imgWebcam)
                startDist = length  # set the initial distance

            length, info, imgWebcam = detector.findDistance(lmList1[8], lmList2[8],
                                                            imgWebcam)  # set the compared distance
            scale = int((length - startDist) // 2)
            cx, cy = info[4:]
            print(scale)

            # sub func3: 显示放缩比例
            volBar = np.interp(scale, [-300, 170], [400, 150])  # 构建映射 [-220, 220]表示scale大致范围 不同图片需要更改 大概是图片size大小
            volPer = np.interp(scale, [-300, 170], [0, 100])  # [400, 150] [0, 100]表示映射成区间范围

            cv2.rectangle(imgWebcam, (50, 150), (85, 400), (0, 140, 255), 3)
            cv2.rectangle(imgWebcam, (50, int(volBar)), (85, 400), (0, 140, 255), cv2.FILLED)
            cv2.putText(imgWebcam, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 140, 255), 3)

    else:
        startDist = None  # if don't detect two hands, update scale strategy

    h1 = 400
    win_size = ((h1 + scale) // 2) * 2  # update img size

    if (cy - win_size // 2) > 0 and (cy + win_size // 2) < 720 and (h1 + scale) >= 2 and (cx - win_size // 2) > 0 and (
            cx + win_size // 2) < 1280:  # 边界条件，约束上下左右及缩小范围

        imgVideo = cv2.resize(imgVideo, (win_size, win_size))

        # draw orange bounding box
        cv2.rectangle(imgWebcam, (cx - win_size // 2, cy - win_size // 2), (cx + win_size // 2, cy + win_size // 2),
                      (0, 140, 255), 4, 1)

        # 框内视频显示 —— 设定候候选框位置，获取ROI
        roi = imgWebcam[cy - win_size // 2:cy + win_size // 2,
              cx - win_size // 2:cx + win_size // 2]  # row range, col range
        # create video mask and inverse mask
        img2gray = cv2.cvtColor(imgVideo, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # black-out the area of video
        imgWebcam_bg = cv2.bitwise_and(roi, roi, mask=mask)
        # take only region of video from video image
        imgVideo_fg = cv2.bitwise_and(imgVideo, imgVideo, mask=mask_inv)
        # put the video in ROI and modify the webcam image
        dst = cv2.add(imgWebcam_bg, imgVideo_fg)

        imgWebcam[cy - win_size // 2:cy + win_size // 2, cx - win_size // 2:cx + win_size // 2] = dst
    else:
        cv2.putText(imgWebcam, str('out of the range'), (5, 30), cv2.FONT_HERSHEY_PLAIN,  # 超边界提示
                    2, (0, 0, 255), 2)

    # func4: calculate fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(imgWebcam, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow('img', imgWebcam)

    frameCounter += 1

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# When everything done, release the capture
cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 删除全部窗口
