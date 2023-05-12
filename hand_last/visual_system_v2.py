import cv2
from PIL import Image
import cvzone
import math
from numpy import *
# from cvzone.HandTrackingModule1 import HandDetector
# from cvzone.HandTrackingModule2 import HandDetector2
from HandTrackingModule1 import HandDetector
from HandTrackingModule2 import HandDetector2
import numpy as np
from MethodUse import readtxt, cv2ImgAddText, fingerStill, inArea, distance
import time
import sys
from yolo import YOLO
from yolo0 import YOLO0

# 加载主界面装饰Logo及模块提示图片
imglogo = cv2.imread('iconImage/logo.png', cv2.IMREAD_UNCHANGED)            # NUDT Logo
imglogo = cv2.resize(imglogo, (80, 80))
img_bbox = cv2.imread('iconImage/bbox2.png', cv2.IMREAD_UNCHANGED)          # 边框 Logo
img_bbox = cv2.resize(img_bbox, (1280, 960))
img_battery = cv2.imread('iconImage/battery2.png', cv2.IMREAD_UNCHANGED)    # 电池 Logo
img_battery = cv2.resize(img_battery, (100, 35))
img_compass = cv2.imread('iconImage/compass.png', cv2.IMREAD_UNCHANGED)     # 指南针 Logo
img_compass = cv2.resize(img_compass, (250, 250))
aim_w, aim_h = 100, 100                                                     # 准心 size
img_aim = cv2.imread('iconImage/aimpoint.png', cv2.IMREAD_UNCHANGED)        # 准心 Logo
img_aim = cv2.resize(img_aim, (aim_w, aim_h))

goforward = cv2.imread('iconImage/forward.png', cv2.IMREAD_UNCHANGED)       # 地图指示 Logo
goforward = cv2.resize(goforward, (100, 200))
dangerpic = cv2.imread('iconImage/danger4.png', cv2.IMREAD_UNCHANGED)       # 预警 Logo
dangerpic = cv2.resize(dangerpic, (130, 130))
img_wait = cv2.imread('iconImage/wait.png', cv2.IMREAD_UNCHANGED)           # 等待 Logo
img_wait = cv2.resize(img_wait, (300, 300))

# 加载目标识别模块对应图片
f15pic = cv2.imread('iconImage/f15.jpg', cv2.IMREAD_UNCHANGED)
f15textpic = cv2.imread('iconImage/f15text.jpg', cv2.IMREAD_UNCHANGED)
b2pic = cv2.imread('iconImage/b2.jpg', cv2.IMREAD_UNCHANGED)
b2textpic = cv2.imread('iconImage/b2text.jpg', cv2.IMREAD_UNCHANGED)
canonpic = cv2.imread('iconImage/cannon.jpg', cv2.IMREAD_UNCHANGED)
canontextpic = cv2.imread('iconImage/canontext.jpg', cv2.IMREAD_UNCHANGED)

# 加载 UI交互 Logo
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
imgmap2 = cv2.imread('iconImage/2.jpg')
img_resize2 = cv2.resize(imgmap2, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)

imgmap21 = cv2.imread('iconImage/2.1.png')
img_resize21 = cv2.resize(imgmap21, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap22 = cv2.imread('iconImage/2.2.png')
img_resize22 = cv2.resize(imgmap22, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap23 = cv2.imread('iconImage/2.3.png')
img_resize23 = cv2.resize(imgmap23, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap3 = cv2.imread('iconImage/3.png')
img_resize3 = cv2.resize(imgmap3, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap4 = cv2.imread('iconImage/4.png')
img_resize4 = cv2.resize(imgmap4, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
imgmap41 = cv2.imread('iconImage/4.1.png')
img_resize41 = cv2.resize(imgmap41, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)


# lmList = []
# yolo = YOLO()
# yolo0 = YOLO0()
# waitt = 1

# center = [400, 300]
# centermap = [700, 300]
# wHalf = 210
# hHalf = 200
# imgmap = cv2.imread('map.jpg')
# resizemap = 0.5
# img_resize = cv2.resize(imgmap, (0, 0), fx=resizemap, fy=resizemap, interpolation=cv2.INTER_NEAREST)
# wHalfmap = 300
# hHalfmap = 200
# startDist0 = None
# scale0 = 0


# frontInter = [20, 25, 35, 45]
# frontSize = [15, 25, 35, 40]
# FrontInter = frontInter[0]
# FrontSize = frontSize[0]
# text_line = [['第一师接到作战命令：进攻A点，', '第一师指派98名士兵参与作战，', '第一师准备在2号公路阻击敌人，', '完成作战任务后，在南城汇合']]

# L = 500
# M = 500
# N = 100
# D = 300
# xita = math.pi / 180 * 60  # 角度xita
# P = 2
# r = 3
# Q = 2.5  # 三点透视参数
# five = math.pi / 180 * 60  # 角度five，三点透视参数
# CenterZ = [1180, 145]
# mulsize = 2
# outl = []

'''
Storedata31 = []
Storedata32 = []
Storedata33 = []
sizesignal = False
movesignal = False
points = [[0, 0, 200, 1], [200, 0, 200, 1], [200, 0, 0, 1], [0, 0, 0, 1],
          [0, 200,  200, 1], [200, 200, 200, 1], [200, 200, 0, 1],
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

# 作战信息参数初始化
center = [400, 300]
centermap = [700, 300]
wHalf, hHalf = 210, 200
wHalfmap, hHalfmap = 300, 200
imgmap = cv2.imread('map.jpg')
resizemap = 0.5
img_resize = cv2.resize(imgmap, (0, 0), fx=resizemap, fy=resizemap, interpolation=cv2.INTER_NEAREST)
startDist0 = None
scale0 = 0

rec = BackgroundRect(center, wHalf, hHalf)
recmap = BackgroundRect(centermap, wHalfmap, hHalfmap)
'''

# 摄像头参数
windoww = 1280
windowh = 960

# 帧率初始化
pTime = 0

# 模型初始化
lmList = []
yolo = YOLO()
yolo0 = YOLO0()
waitt = 1   # 全局预警卡顿提示

# 界面模式/按键初始化
mode = 'home'
iconx = 400
backIcon = True
backIcon2 = True
backPosition = [100, 200, 400, 500]

# load 主摄像头视频
cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1.set(6,cv2.VideoWriter.fourcc('M','J','P','G'))
cap1.set(3, windoww)
cap1.set(4, windowh)
detector = HandDetector(detectionCon=0.8, maxHands=2)
 
# load 第二摄像头视频
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap2.set(6,cv2.VideoWriter.fourcc('M','J','P','G'))
cap2.set(3, windoww)
cap2.set(4, windowh)
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector2 = HandDetector2(detectionCon=0.8, maxHands=2)

# delayBack = 0  # 返回防误触，防止连续点击两次返回

while True:
    Rate1 = 1
    success1, img = cap1.read()
    # img = cv2.flip(img, 1)
    allHands, img = detector.findHands(img, draw=True, flipType=True)  # flipType=False
    l = 200
    if len(allHands) != 0:
        lmList = allHands[0]['lmList']
        cursor = lmList[8]
        # 食指添加瞄准镜准心
        if cursor[0] - aim_w // 2 > 0 and cursor[0] + aim_w // 2 < 1280 and cursor[1] - aim_h // 2 > 0 and cursor[1] + aim_h // 2 < 960:
            img = cvzone.overlayPNG(img, img_aim, [cursor[0] - aim_w // 2, cursor[1] - aim_h // 2])

        l, _, _ = detector.findDistance(lmList[8], lmList[12], img)

        # 辅助检测模块[返回键]防误触
        if not backIcon and not (iconx + 150 < cursor[0] < iconx + 250 or 550 < cursor[1] < 650):
            backIcon = not backIcon
        # 视频指令模块[返回键]防误触
        if not backIcon2 and not (100 < cursor[0] < 200 or 400 < cursor[1] < 500):
            backIcon2 = not backIcon2

    if mode == 'home':
        if len(allHands) != 0:
            lmList = allHands[0]['lmList']
            cursor = lmList[8]
            l, _, _ = detector.findDistance(lmList[8], lmList[12], img)
            if l < 50 and iconx + 200 < cursor[0] < iconx + 300 and 400 < cursor[1] < 500:
                Rate1 = 1
                mode = 'function'
            else:
                blk = np.zeros(img.shape, np.uint8)
                blk[400:500, iconx + 200:iconx + 300] = img_resize0[:, :, :]
                img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)
                img = cv2ImgAddText(img, '主菜单', iconx + 215, 505)   # 颜色和大小采用默认值
        else:
            blk = np.zeros(img.shape, np.uint8)
            blk[400:500, iconx + 200:iconx + 300] = img_resize0[:, :, :]
            img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)
            img = cv2ImgAddText(img, '主菜单', iconx + 215, 505)

    if mode == 'function':
        if l < 50 and iconx + 150 < cursor[0] < iconx + 250 and 550 < cursor[1] < 650 and backIcon:
            mode = 'home'
            backIcon = not backIcon

        elif l < 50 and iconx < cursor[0] < iconx + 100 and 550 < cursor[1] < 650:
            mode = 'detect_pic'
            # initialize parameter
            detect_part_count = 0
            points = [[0, 0, 200, 1], [200, 0, 200, 1], [200, 0, 0, 1], [0, 0, 0, 1],
                      [0, 200, 200, 1], [200, 200, 200, 1], [200, 200, 0, 1],
                      [0, 200, 0, 1]]

            # 3D 立方体
            L, M, N, D = 500, 500, 100, 300
            xita = math.pi / 180 * 60           # 角度xita
            P = 2
            r = 3
            Q = 2.5                             # 三点透视参数
            five = math.pi / 180 * 60           # 角度five，三点透视参数
            CenterZ = [1180, 145]
            mulsize = 2
            outl = []

            # 返回防误触，防止目标3D模型子功能连续点击两次返回
            delayBack = 0

        elif l < 50 and iconx < cursor[0] < iconx + 100 and 400 < cursor[1] < 500:
            mode = 'Showway'    # 行进位置
            # initialize parameter
            cx, cy = 1015, 315                  # 初始视频中心位置坐标
            win_size = 400                      # 初始视频窗大小
            startDist = None                    # 初始放缩参考距离
            scale = 0                           # 初始放缩比例
            frameCounter = 0                    # 循环播放视频帧
            myVid = cv2.VideoCapture('move.mp4')

            # 初始化 作战记录 子模块（键盘）参数
            # myEquation = ''
            # delayCounter = 0

            # 初始化 作战信息 子模块（文本、图片）参数
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

            center = [400, 300]
            centermap = [700, 300]
            wHalf, hHalf = 210, 200
            wHalfmap, hHalfmap = 300, 200
            imgmap = cv2.imread('map.jpg')
            resizemap = 0.5
            img_resize = cv2.resize(imgmap, (0, 0), fx=resizemap, fy=resizemap, interpolation=cv2.INTER_NEAREST)
            startDist0 = None
            scale0 = 0

            rec = BackgroundRect(center, wHalf, hHalf)
            recmap = BackgroundRect(centermap, wHalfmap, hHalfmap)

            frontInter = [20, 25, 35, 45]
            frontSize = [15, 25, 35, 40]
            FrontInter = frontInter[0]
            FrontSize = frontSize[0]
            text_line = [['第一师接到作战命令：进攻A点，', '第一师指派98名士兵参与作战，', '第一师准备在2号公路阻击敌人，', '完成作战任务后，在南城汇合']]

        elif l < 50 and iconx < cursor[0] < iconx + 100 and 250 < cursor[1] < 350:
            mode = 'Facevideo'  # 视频指令
            # initialize parameter
            cx, cy = 1030, 250                  # 第三方视频中心位置坐标
            scaleface = 0                       # 初始放缩比例
            startDistface = None                # 初始放缩参考距离
            win_size_w, win_size_h = 424, 240   # 初始视频窗大小
            cap2.set(3, win_size_w)
            cap2.set(4, win_size_h)

        elif l < 50 and iconx < cursor[0] < iconx + 100 and 100 < cursor[1] < 200:
            mode = 'RemoteTrack'
            detecT = False
            clickt = False

        else:
            blk = np.zeros(img.shape, np.uint8)
            blk[550:650, iconx + 150:iconx + 250] = img_resize00[:, :, :]
            blk[550:650, iconx:iconx + 100] = img_resize1[:, :, :]
            img = cv2ImgAddText(img, '辅助检测', iconx + 5, 655)   # 颜色、大小缺省，采用默认值
            blk[400:500, iconx:iconx + 100] = img_resize23[:, :, :]
            img = cv2ImgAddText(img, '作战导航', iconx + 5, 505)
            blk[250:350, iconx:iconx + 100] = img_resize3[:, :, :]
            img = cv2ImgAddText(img, '视频指令', iconx + 5, 355)
            blk[100:200, iconx:iconx+100] = img_resize4[:, :, :]
            img = cv2ImgAddText(img, '远程态势', iconx + 5, 205)
            img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)

    if mode == 'detect_pic':
        if l < 50 and iconx + 150 < cursor[0] < iconx + 250 and 550 < cursor[1] < 650 and backIcon:
            mode = 'function'
            backIcon = not backIcon
        elif l < 50 and iconx < cursor[0] < iconx + 100 and 400 < cursor[1] < 500:
            mode = 'detect_all'
            if waitt == 1:
                img = cvzone.overlayPNG(img, img_wait, [500, 300])
        elif l < 50 and iconx < cursor[0] < iconx + 100 and 250 < cursor[1] < 350:
            mode = 'detect_part'
            # delayBack = 0   # 返回防误触，防止目标3D模型子功能连续点击两次返回
        else:
            blk = np.zeros(img.shape, np.uint8)
            blk[550:650, iconx + 150:iconx + 250] = img_resize00[:, :, :]
            blk[400:500, iconx:iconx + 100] = img_resize11[:, :, :]
            img = cv2ImgAddText(img, '全局预警', iconx + 5, 505)       # 颜色、大小缺省，采用默认值
            blk[250:350, iconx:iconx + 100] = img_resize12[:, :, :]
            img = cv2ImgAddText(img, '目标识别', iconx + 5, 355)
            img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)

    if mode == 'Showway':
        # 初始化 作战记录（键盘）参数
        myEquation = ''
        delayCounter = 0

        if l < 50 and backPosition[0] < cursor[0] < backPosition[1] and backPosition[2] < cursor[1] < backPosition[3] and backIcon2:
            mode = 'function'
            backIcon2 = not backIcon2
        if l < 50 and 100 < cursor[0] < 200 and 100 < cursor[1] < 200:  # 作战信息
            mode = 'Remessage_read'
        if l < 50 and 100 < cursor[0] < 200 and 250 < cursor[1] < 350:  # 发送指令
            mode = 'Sendmessage'
            replymes = ''
            replyCount = 0
            hint_success = 20    # 消息发送成功通知

        # 循环播放视频
        if frameCounter == myVid.get(cv2.CAP_PROP_FRAME_COUNT):  # 获取总帧数，如果计数帧=总帧数，设置0帧重新播放视频
            myVid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        ret2, imgVideo = myVid.read()

        if len(allHands) == 1:      # 移动视频
            lmList = allHands[0]['lmList']
            length, info, img = detector.findDistance(lmList[8], lmList[12], img)
            if length < 60:
                cursor = lmList[8]
                if cx - win_size // 2 < cursor[0] < cx + win_size // 2 and cy - win_size // 2 < cursor[1] < cy + win_size // 2:
                    cx, cy = cursor[0], cursor[1]

        if len(allHands) == 2:      # 移动+放缩视频
            if detector.fingersUp(allHands[0]) == [0, 1, 0, 0, 0] and \
                    detector.fingersUp(allHands[1]) == [0, 1, 0, 0, 0]:
                lmList1 = allHands[0]['lmList']
                lmList2 = allHands[1]['lmList']
                if startDist is None:
                    length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
                    startDist = length

                length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
                scale = int((length - startDist) // 2)
                cx, cy = info[4:]
                # print(scale)
                # 放缩比例条
                volBar = np.interp(scale, [-300, 170], [500, 150])
                volPer = np.interp(scale, [-300, 170], [0, 100])

                cv2.rectangle(img, (300, 150), (325, 500), (140, 199, 0), 3)    # 土耳其玉色
                cv2.rectangle(img, (300, int(volBar)), (325, 500), (140, 199, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)} %', (290, 530), cv2.FONT_HERSHEY_COMPLEX, 0.75, (140, 199, 0), 2)
        else:
            startDist = None

        h1 = 400
        win_size = ((h1 + scale) // 2) * 2  # update img size

        if (cy - win_size // 2) > 0 and (cy + win_size // 2) < 960 and (h1 + scale) >= 2 and (  # 边界条件
                cx - win_size // 2) > 0 and (cx + win_size // 2) < 1280:

            imgVideo = cv2.resize(imgVideo, (win_size, win_size))

            # 视频边框（土耳其玉色）
            cv2.rectangle(img, (cx - win_size // 2, cy - win_size // 2), (cx + win_size // 2, cy + win_size // 2), (140, 199, 0), 4, 1)

            # write location slogan
            cv2.putText(img, f'Location', (cx + win_size // 2 - 125, cy - win_size // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.circle(img, (cx + win_size // 2 - 12, cy - win_size // 2 - 16), 11, (140, 199, 0), -1)

            # 第三方视频流与主摄像头逻辑运算
            roi = img[cy - win_size // 2:cy + win_size // 2, cx - win_size // 2:cx + win_size // 2]
            img2gray = cv2.cvtColor(imgVideo, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            imgWebcam_bg = cv2.bitwise_and(roi, roi, mask=mask)
            imgVideo_fg = cv2.bitwise_and(imgVideo, imgVideo, mask=mask_inv)
            dst = cv2.add(imgWebcam_bg, imgVideo_fg)

            img[cy - win_size // 2:cy + win_size // 2, cx - win_size // 2:cx + win_size // 2] = dst
        else:
            cv2.putText(img, str('It\'s out of range!'), (120, 70), cv2.FONT_HERSHEY_SIMPLEX,  # 超边界提示
                        1.1, (0, 0, 255), 3)
        frameCounter += 1   # 循环计数

        blk = np.zeros(img.shape, np.uint8)
        blk[100:200, backPosition[0]:backPosition[1]] = img_resize2[:, :, :]
        img = cv2ImgAddText(img, '作战信息', 105, 205)   # 颜色、大小缺省，采用默认值
        blk[250:350, backPosition[0]:backPosition[1]] = img_resize22[:, :, :]
        img = cv2ImgAddText(img, '作战记录', 105, 355)
        blk[backPosition[2]:backPosition[3], backPosition[0]:backPosition[1]] = img_resize00[:, :, :]
        img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)
        img = cvzone.overlayPNG(img, goforward, [650, 100])

    if mode == 'Facevideo':
        if l < 50 and backPosition[0] < cursor[0] < backPosition[1] and backPosition[2] < cursor[1] < backPosition[3] and backIcon2:
            mode = 'function'
            backIcon2 = not backIcon2
        success_2, img_2 = cap2.read()
        img_2 = cv2.flip(img_2, 1)
        # print(img_2.shape)

        if len(allHands) == 1:      # 移动视频
            lmList = allHands[0]['lmList']
            length, info, img = detector.findDistance(lmList[8], lmList[12], img)
            if length < 55:
                cursor = lmList[8]
                if cx - win_size_w // 2 < cursor[0] < cx + win_size_w // 2 and cy - win_size_h // 2 < cursor[1] < cy + win_size_h // 2:
                    cx, cy = cursor[0], cursor[1]

        if len(allHands) == 2:      # 移动+放缩视频
            if detector.fingersUp(allHands[0]) == [0, 1, 0, 0, 0] and \
                    detector.fingersUp(allHands[1]) == [0, 1, 0, 0, 0]:                 # [1, 1, 0, 0, 0]
                lmList1 = allHands[0]['lmList']
                lmList2 = allHands[1]['lmList']
                if startDistface is None:
                    length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
                    startDistface = length                                              # set the initial distance

                length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # set the compared distance
                scaleface = int((length - startDistface) // 2)
                cx, cy = info[4:]

                # 放缩比例条
                volBar = np.interp(scaleface, [-300, 170], [500, 150])
                volPer = np.interp(scaleface, [-300, 170], [0, 100])

                cv2.rectangle(img, (300, 150), (325, 500), (0, 140, 255), 3)
                cv2.rectangle(img, (300, int(volBar)), (325, 500), (0, 140, 255), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)} %', (290, 530), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 140, 255), 2)
        else:
            startDistface = None                                                        # if don't detect two hands, update scale strategy

        w, h = 424, 240     # 固定比例
        win_size_w = ((w + scaleface) // 2) * 2
        win_size_h = (int((h / w * win_size_w)) // 2) * 2

        if (cy - win_size_h // 2) > 0 and (cy + win_size_h // 2) < 960 and (h + scaleface) >= 2 and (w + scaleface) >= 2 and (
                cx - win_size_w // 2) > 0 and (cx + win_size_w // 2) < 1280:            # 边界条件，约束上下左右及缩小范围

            img_2 = cv2.resize(img_2, (win_size_w, win_size_h))

            # 视频边框（橙色）
            cv2.rectangle(img, (cx - win_size_w // 2, cy - win_size_h // 2), (cx + win_size_w // 2, cy + win_size_h // 2), (0, 140, 255), 4, 1)
            # write LIVE slogan
            cv2.putText(img, f'L I V E', (cx + win_size_w // 2 - 115, cy - win_size_h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3)
            cv2.circle(img, (cx + win_size_w // 2 - 12, cy - win_size_h // 2 - 18), 11, (0, 140, 255), -1)

            # 第三方视频流与主摄像头逻辑运算
            roi = img[cy - win_size_h // 2:cy + win_size_h // 2, cx - win_size_w // 2:cx + win_size_w // 2]
            img2gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 254, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)
            imgWebcam_bg = cv2.bitwise_and(roi, roi, mask=mask)
            imgVideo_fg = cv2.bitwise_and(img_2, img_2, mask=mask_inv)
            dst = cv2.add(imgWebcam_bg, imgVideo_fg)

            img[cy - win_size_h // 2:cy + win_size_h // 2, cx - win_size_w // 2:cx + win_size_w // 2] = dst
        else:
            cv2.putText(img, str('It\'s out of range!'), (120, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        blk = np.zeros(img.shape, np.uint8)
        blk[backPosition[2]:backPosition[3], backPosition[0]:backPosition[1]] = img_resize00[:, :, :]
        img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)

    if mode == 'RemoteTrack':
        if l < 50 and iconx + 150 < cursor[0] < iconx + 250 and 550 < cursor[1] < 650 and backIcon:
            mode = 'function'
            backIcon = not backIcon
        elif l < 50 and iconx < cursor[0] < iconx + 100 and 250 < cursor[1] < 350:
            mode = 'RemoteTrack_duoji'
        else:
            blk = np.zeros(img.shape, np.uint8)
            blk[550:650, iconx+150:iconx+250] = img_resize00[:, :, :]
            blk[250:350, iconx:iconx + 100] = img_resize41[:, :, :]
            img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)

    if mode == 'detect_part':       # 局部检测
        if len(allHands) != 0:
            cursor = lmList[8]
            m_b, _, _ = detector.findDistance(lmList[4], lmList[8], img)
            if l < 50:
                win_sizede = 320    # 截取框大小
                if cursor[0] > win_sizede and cursor[1] > win_sizede:
                    box_left = cursor[0] - win_sizede
                    box_top = cursor[1] - win_sizede
                    cv2.rectangle(img, (box_left, box_top), cursor, (0, 255, 0), 2)         # 绿框
                    snip_img = img[box_top:cursor[1], box_left:cursor[0]]

                    if m_b < 50:    # 开始检测
                        cv2.rectangle(img, (box_left, box_top), cursor, (255, 255, 0), 2)   # 蓝框
                        cv2.imwrite('snip.jpg', snip_img)
                        img_temp = Image.open('snip.jpg')
                        res_image, out_class = yolo.detect_image(img_temp)
                        cv2.putText(img, str(out_class), (box_left, box_top), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                        print(out_class)

                        detect_part_count += 1      # 延时：展示目标类别信息
                        if detect_part_count > 10:
                            if out_class[0] == 'FIGHTER_F15A' or out_class[0] == 'FIGHTER_F18E':
                                mode = 'f15_obj'
                            if out_class[0] == 'BOMBER_B2':
                                mode = 'B2_obj'
                            if out_class[0] == 'CANNON_jorge' or out_class[0] == 'CANNON_PLZ05' or out_class[0] == 'CANNON_high':
                                mode = 'cannon_infor'
                else:
                    cv2.putText(img, str('It\'s out of range!'), (120, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

            # 返回防误触，防止目标3D模型子功能模式下连续点击两次返回
            if delayBack != 0:
                delayBack += 1
                if delayBack > 10:
                    delayBack = 0

            if l < 50 and delayBack == 0 and backPosition[0] < cursor[0] < backPosition[1] and backPosition[2] < cursor[1] < backPosition[3] and backIcon:
                mode = 'detect_pic'
                backIcon = not backIcon

        blk = np.zeros(img.shape, np.uint8)
        blk[backPosition[2]:backPosition[3], backPosition[0]:backPosition[1]] = img_resize00[:, :, :]
        img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)

    if mode == 'detect_all':
        carnum = 0
        aernum = 0
        personnum = 0
        trucknum = 0
        if waitt == 1:
            waitt -= 1
        else:
            if l < 50 and backPosition[0] < cursor[0] < backPosition[1] and 250 < cursor[1] < 350:
                mode = 'detect_part'

            if len(allHands) != 0:
                if l < 50 and backPosition[0] < cursor[0] < backPosition[1] and backPosition[2] < cursor[1] < backPosition[3] and backIcon:
                    mode = 'detect_pic'
                    backIcon = not backIcon

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(np.uint8(img))
            img, out_class_danger = yolo0.detect_image(img)
            img = np.array(img)
            print(out_class_danger)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if 'nothing' not in out_class_danger:       # nothing 逻辑？
                if 'car' in out_class_danger or 'aeroplane' in out_class_danger or 'person' in out_class_danger or 'truck' in out_class_danger:
                    carnum = out_class_danger.count('car')
                    aernum = out_class_danger.count('aeroplane')
                    personnum = out_class_danger.count('person')
                    trucknum = out_class_danger.count('truck')
                    # 预警提示
                    if carnum != 0 or aernum != 0:
                        img = cvzone.overlayPNG(img, dangerpic, [560, 60])  # 预警 Logo
                        cv2.putText(img, str('Danger!'), (565, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # 选定类别/数量统计
            cv2.putText(img, ('person: ' + str(personnum)), (1060, 110), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(img, ('car: ' + str(carnum)), (1120, 140), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(img, ('plane: ' + str(aernum)), (1088, 170), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.putText(img, ('trunck: ' + str(trucknum)), (1064, 200), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        blk = np.zeros(img.shape, np.uint8)
        blk[backPosition[2]:backPosition[3], backPosition[0]:backPosition[1]] = img_resize00[:, :, :]
        blk[250:350, backPosition[0]:backPosition[1]] = img_resize12[:, :, :]
        img = cv2ImgAddText(img, '目标识别', backPosition[0] + 5, 355)     # 颜色、大小缺省，采用默认值
        img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)

    if mode == 'f15_obj' or mode == 'B2_obj' or mode == 'cannon_infor':             # 方便扩展增加
        if l < 50 and backPosition[0] < cursor[0] < backPosition[1] and backPosition[2] < cursor[1] < backPosition[3] and backIcon2:
            mode = 'detect_pic'     # 精简显示目标参数/图片，若点击返回，直接返回到检测模式，需重新选择粗检/精检
            detect_part_count = 0
            backIcon2 = not backIcon2

        # 初始化移动/放缩 F15 参数
        if mode == 'f15_obj':
            img_model = cv2.imread('iconImage/j15_3d.png', cv2.IMREAD_UNCHANGED)    # 初始化显示图片
            img_model_3d = cv2.resize(img_model, (400, 350))
            newH, newW, _ = img_model_3d.shape
            # print(newW, newH)
        # 初始化移动/放缩 B20 参数
        elif mode == 'B2_obj':
            img_model = cv2.imread('iconImage/b20_3d.png', cv2.IMREAD_UNCHANGED)
            img_model_3d = cv2.resize(img_model, (400, 350))
            newH, newW, _ = img_model_3d.shape
        # 初始化移动/放缩 炮车 参数
        elif mode == 'cannon_infor':
            img_model = cv2.imread('iconImage/cannon_3d.png', cv2.IMREAD_UNCHANGED)
            img_model_3d = cv2.resize(img_model, (400, 200))
            newH, newW, _ = img_model_3d.shape

        # 进入移动/放缩模块
        if l < 50 and 100 < cursor[0] < 200 and 250 < cursor[1] < 350:
            mode = 'Remessage_3D'
            scale = 0
            startDist = None
            cx, cy = 630, 460   # 初始显示位置

        # 目标参数/图片查看
        if mode == 'f15_obj':
            img[200:500, 710:1110] = f15pic[:, :, :]
            blk = np.zeros(img.shape, np.uint8)
            img[200:547, 300:700] = f15textpic[:, :, :]
        elif mode == 'B2_obj':
            img[200:500, 710:1110] = b2pic[:, :, :]
            blk = np.zeros(img.shape, np.uint8)
            img[200:533, 300:700] = b2textpic[:, :, :]
        elif mode == 'cannon_infor':
            img[200:500, 710:1110] = canonpic[:, :, :]
            blk = np.zeros(img.shape, np.uint8)
            img[200:543, 300:700] = canontextpic[:, :, :]

        blk[250:350, backPosition[0]:backPosition[1]] = img_resize21[:, :, :]
        img = cv2ImgAddText(img, '3D模型', 110, 355)  # 颜色、大小缺省，采用默认值
        blk[backPosition[2]:backPosition[3], backPosition[0]:backPosition[1]] = img_resize00[:, :, :]
        img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)

    if mode == 'Remessage_read':
        RecLeftup = (rec.center[0] - rec.w, rec.center[1] - rec.h)
        RecRightdown = (rec.center[0] + rec.w, rec.center[1] + rec.h)
        RecLeftupmap = (int(recmap.center[0] - recmap.w), int(recmap.center[1] - recmap.h))
        RecRightdownmap = (int(recmap.center[0] + recmap.w), int(recmap.center[1] + recmap.h))

        if len(allHands) != 0:
            if l < 50 and backPosition[0] < cursor[0] < backPosition[1] and backPosition[2] < cursor[1] < backPosition[3] and backIcon2:
                mode = 'Showway'
                backIcon2 = not backIcon2

            if len(allHands) == 2:  # 双手
                if allHands[0]['type'] == 'Left':
                    lmListLeft = allHands[0]['lmList']
                    lmListRight = allHands[1]['lmList']
                else:
                    lmListLeft = allHands[1]['lmList']
                    lmListRight = allHands[0]['lmList']

                Lindexfin = lmListLeft[8]
                Rindexfin = lmListRight[8]
                Lindexfinger = inArea(lmListLeft[8], RecLeftup, RecRightdown)
                Rindexfinger = inArea(lmListRight[8], RecLeftup, RecRightdown)
                Lindexfingermap = inArea(lmListLeft[8], RecLeftupmap, RecRightdownmap)
                Rindexfingermap = inArea(lmListRight[8], RecLeftupmap, RecRightdownmap)

                updetect1 = detector.fingersUp(allHands[1]) == [0, 1, 0, 0, 0]
                updetect2 = detector.fingersUp(allHands[0]) == [0, 1, 0, 0, 0]

                if updetect1 and updetect2 and Lindexfinger and Rindexfinger:
                    whalf = int(distance(Lindexfin, Rindexfin))
                    hhalf = int(distance(Lindexfin, Rindexfin))
                    if 250 <= whalf <= 800:
                        if 250 <= whalf <= 300:
                            FrontInter = frontInter[1]
                            FrontSize = frontSize[1]
                        elif 300 < whalf <= 450:
                            FrontInter = frontInter[2]
                            FrontSize = frontSize[2]
                        elif 450 < whalf <= 800:
                            FrontInter = frontInter[3]
                            FrontSize = frontSize[3]
                        cxrec = int((Lindexfin[0] + Rindexfin[0])/2)
                        cyrec = int((Lindexfin[1] + Rindexfin[1])/2)
                        rec.update((cxrec, cyrec))
                        rec.updateSize(whalf, hhalf)
                    elif whalf < 200:
                        FrontInter = frontInter[0]
                        FrontSize = frontSize[0]
                        whalf = 180
                        hhalf = 180
                        cxrec = int((Lindexfin[0] + Rindexfin[0]) / 2)
                        cyrec = int((Lindexfin[1] + Rindexfin[1]) / 2)
                        rec.update((cxrec, cyrec))
                        rec.updateSize(whalf, hhalf)

                if updetect1 and updetect2 and Lindexfingermap and Rindexfingermap:
                    cxrec = int((Lindexfin[0] + Rindexfin[0]) / 2)
                    cyrec = int((Lindexfin[1] + Rindexfin[1]) / 2)
                    distancemap = int(distance(Lindexfin, Rindexfin))
                    if startDist0 is None:
                        startDist0 = distancemap
                    scale0 = int((distancemap - startDist0) // 2)

                    h1, w1, _ = cv2.resize(imgmap, (0, 0), fx=resizemap, fy=resizemap, interpolation=cv2.INTER_NEAREST).shape
                    newH, newW = ((h1 + scale0) // 2) * 2, ((w1 + scale0) // 2) * 2
                    img_resize = cv2.resize(imgmap, (newW, newH))

                    recmap.update((cxrec, cyrec))
                    recmap.updateSize(newW/2, newH/2)
                    RecLeftupmap = (int(recmap.center[0] - recmap.w), int(recmap.center[1] - recmap.h))
                    RecRightdownmap = (int(recmap.center[0] + recmap.w), int(recmap.center[1] + recmap.h))
                else:
                    startDist0 = None

            else:
                lmList = allHands[0]['lmList']
                finger8 = lmList[8]
                rec8 = inArea(lmList[8], RecLeftup, RecRightdown)
                rec12 = inArea(lmList[12], RecLeftup, RecRightdown)
                recmap8 = inArea(lmList[8], RecLeftupmap, RecRightdownmap)
                recmap12 = inArea(lmList[12], RecLeftupmap, RecRightdownmap)

                Indesfingermap = inArea(lmList[8], RecLeftupmap, RecRightdownmap)
                if l < 50 and rec8 and rec12:
                    rec.update(finger8)
                if l < 50 and recmap8 and recmap12 and not (rec8 and rec12):
                    recmap.update(finger8)

        xi = RecLeftup[0]
        yi = RecLeftup[1] - 25
        blk = np.zeros(img.shape, np.uint8)
        cv2.rectangle(blk, RecLeftup, RecRightdown, (255, 255, 255), -1)
        img = cv2.addWeighted(img, 1.0, blk, 0.8, 1)

        if len(allHands) == 2 and Lindexfinger and Rindexfinger:
                cv2.circle(img, Lindexfin, 15, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, Rindexfin, 15, (0, 255, 0), cv2.FILLED)

        readF = text_line[0]
        for i, text in enumerate(readF):
            if text:
                img = cv2ImgAddText(img, text, xi, yi + FrontInter * (i + 1) + 15, (0, 0, 139), FrontSize)

        if inArea(RecLeftupmap, (0, 0), (1250, 900)) and inArea(RecRightdownmap, (0, 0), (1250, 900)) and scale < 450:
            img[RecLeftupmap[1]:RecRightdownmap[1], RecLeftupmap[0]:RecRightdownmap[0]] = img_resize[:, :, :]
        else:
            scale0 = 0
            img_resize = cv2.resize(imgmap, (0, 0), fx=resizemap, fy=resizemap, interpolation=cv2.INTER_NEAREST)
            recmap = BackgroundRect(centermap, wHalfmap, hHalfmap)
            cv2.putText(img, str('It\'s out of range!'), (120, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        blk = np.zeros(img.shape, np.uint8)
        blk[backPosition[2]:backPosition[3], backPosition[0]:backPosition[1]] = img_resize00[:, :, :]
        img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)

    if mode == 'Sendmessage':
        class Button():
            def __init__(self, pos, text, size=(70, 70)):
                self.pos = pos
                self.text = text
                self.size = size

            def draw(self, img, color=(33, 36, 41)):     # 键盘背景色 (250, 51, 153) 紫色
                x, y = self.pos
                w, h = self.size
                cv2.rectangle(img, self.pos, (x + w, y + h), color, cv2.FILLED)
                cv2.putText(img, self.text, (x + 20, y + 35), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                return img

        keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", 'P', "[", ']'],
                ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'", "Enter"],
                ["Z", "X", "C", "V", "B", "N", 'M', ",", ".", "/", "Backspace"]]
        buttonList = []
        interdistance = 80
        '''延时显示对话框'''
        if replyCount < 20:
            replyCount += 1
        else:pass
        if l < 50 and backPosition[0] < cursor[0] < backPosition[1] and backPosition[2] < cursor[1] < backPosition[3] and backIcon2:
            mode = 'Showway'
            backIcon2 = not backIcon2

        for i in range(len(keys)):
            for j, key in enumerate(keys[i]):
                if i == 2 and j == 10:
                    buttonList.append(
                        Button([interdistance * j + 200 + i * 12, 450 + interdistance * i], key, size=[150, 70]))
                elif i == 1 and j == 11:
                    buttonList.append(
                        Button([interdistance * j + 200 + i * 12, 450 + interdistance * i], key, size=[100, 70]))
                else:
                    buttonList.append(Button([interdistance * j + 200 + i * 12, 450 + interdistance * i], key))

        def AllDraw(img):
            for btn in buttonList:
                img = btn.draw(img)
            return img

        img_ori = img.copy()    # 复制原始图像，以便设置透明度
        img = AllDraw(img)
        length, _, img = detector.findDistance(lmList[4], lmList[12], img)

        for i, button in enumerate(buttonList):
            x, y = button.pos
            w, h = button.size
            deviation = 1
            if x - deviation < lmList[8][0] < x + w + deviation \
                    and y - deviation < lmList[8][1] < y + h + deviation:
                img = button.draw(img, (100, 150, 0))
                if length < 50 and delayCounter == 0:
                    Clickvalue = keys[int(i // 12)][int(i % 12)]
                    if Clickvalue == 'Backspace' and myEquation != '':
                        myEquation = myEquation[:len(myEquation) - 1]
                    elif Clickvalue == 'Backspace' and myEquation == '':
                        pass
                    # elif Clickvalue == '---':   # 空格
                    #     myEquation += ' '
                    elif Clickvalue == 'Enter':   # 空格
                        replymes = myEquation
                        myEquation = ''
                    else:
                        myEquation += Clickvalue
                    delayCounter = 1

        if delayCounter != 0:
            delayCounter += 1
            if delayCounter > 10:
                delayCounter = 0

        cv2.rectangle(img, (300, 100), (700, 250), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, 'Message:', (300, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (70, 23, 11), 3)
        cv2.putText(img, myEquation, (300, 170), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 144, 30), 3)
        if replyCount == 20:
            cv2.rectangle(img, (900, 100), (1150, 400), (33, 36, 41), cv2.FILLED)
            cv2.putText(img, 'Commander:', (900, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 144, 30), 2)
            cv2.putText(img, 'Come to A please!', (900, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 248, 248), 2)
            cv2.putText(img, 'Me:', (1100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 144, 30), 2)
            cv2.putText(img, replymes, (1100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 248, 248), 2)

        if replymes == 'OK' and hint_success > 0:
            cv2.putText(img, 'Send Successfully!', (915, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 144, 30), 2)
            hint_success -= 1

        # 设置键盘透明度
        alpha = 0.2
        mask = img.astype(bool)
        img_ori[mask] = cv2.addWeighted(img_ori, alpha, img, 1 - alpha, 0)[mask]  # fuse img
        img = img_ori

        blk = np.zeros(img.shape, np.uint8)
        blk[backPosition[2]:backPosition[3], backPosition[0]:backPosition[1]] = img_resize00[:, :, :]
        img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)

    if mode == 'Remessage_3D':
        out0 = []
        Sumpoint = 0
        img_model = img_model_3d

        if len(allHands) != 0:
            if l < 50 and backPosition[0] < cursor[0] < backPosition[1] and backPosition[2] < cursor[1] < backPosition[3] and backIcon2:
                mode = 'detect_part'
                backIcon2 = not backIcon2   # 精检显示目标3D模型，若点击返回，则返回到上一级（精检）
                delayBack = 1               # 避免误触（同一位置连续两次点击返回），添加延迟约束防误触

            # lmList = allHands[0]['lmList']
            Are = 150
            RU = [CenterZ[0] - Are, CenterZ[1] - Are]
            RD = [CenterZ[0] + Are, CenterZ[1] + Are]

            if len(allHands) == 1:
                lmList1 = allHands[0]['lmList']
                # Check if clicked
                length, info, img = detector.findDistance(lmList1[8], lmList1[12], img)
                # print(length)
                if length < 60:
                    cursor = lmList1[8]
                    # Check if in region
                    if cx - newW // 2 < cursor[0] < cx + newW // 2 and cy - newH // 2 < cursor[1] < cy + newH // 2:
                        cx, cy = cursor[0], cursor[1]

            if len(allHands) == 2:
                lmList1 = allHands[0]['lmList']
                lmList2 = allHands[1]['lmList']
                if detector.fingersUp(allHands[0]) == [0, 1, 0, 0, 0] and \
                        detector.fingersUp(allHands[1]) == [0, 1, 0, 0, 0]:
                    if startDist is None:
                        length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
                        startDist = length                                                  # set the initial distance

                    length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)  # set the compared distance
                    scale = int((length - startDist) // 2)
                    cx, cy = info[4:]

                    # sub func3: 显示放缩比例
                    volBar = np.interp(scale, [-220, 220], [600, 250])
                    volPer = np.interp(scale, [-220, 220], [0, 100])

                    cv2.rectangle(img, (1050, 250), (1075, 600), (214, 112, 218), 3)
                    cv2.rectangle(img, (1050, int(volBar)), (1075, 600), (214, 112, 218), cv2.FILLED)
                    cv2.putText(img, f'{int(volPer)} %', (1040, 630), cv2.FONT_HERSHEY_COMPLEX, 0.75, (214, 112, 218), 2)
            else:
                startDist = None  # if don't detect two hands, update scale strategy

            mm, _, _ = detector.findDistance(lmList1[4], lmList1[8], img)
            theta = mm * 360 / 280
            cosr = np.cos(theta * np.pi / 180)
            sinr = np.sin(theta * np.pi / 180)
            Rx = np.array([[1, 0, 0, 0], [0, cosr, -sinr, 0], [0, sinr, cosr, 0], [0, 0, 0, 1]])
            Ry = np.array([[cosr, 0, sinr, 0], [0, 1, 0, 0], [-sinr, 0, cosr, 0], [0, 0, 0, 1]])
            Rz = np.array([[cosr, -sinr, 0, 800], [sinr, cosr, 0, 300], [0, 0, 1, 0], [0, 0, 0, 1]])
            mtranslation = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [L, M, N, 1]], dtype='float32')

            # 沿y轴旋转
            mrotatey = np.array([[math.cos(xita), 0, -math.sin(xita), 0], [0, 1, 0, 0],
                                 [math.sin(xita), 0, math.cos(xita), 0], [0, 0, 0, 1]],
                                dtype='float32')
            # 沿x轴旋转
            mrotatex = np.array([[1, 0, 0, 0], [0, math.cos(five), math.sin(five), 0],
                                [0, -math.sin(five), math.cos(five), 0], [0, 0, 0, 1]],
                                dtype='float32')
            # 透视
            mperspective = np.array([[1, 0, 0, P], [0, 1, 0, Q], [0, 0, 1, r], [0, 0, 0, 1]], dtype='float32')
            # 向xoy平面做正投影
            mprojectionxy = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype='float32')
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
            points1 = np.array([tuple(outl[0]), tuple(outl[1]), tuple(outl[2]), tuple(outl[3])])
            points2 = np.array([tuple(outl[4]), tuple(outl[5]), tuple(outl[6]), tuple(outl[7])])
            img = cv2.fillPoly(img, [points1], [100, 0, 250])
            img = cv2.fillPoly(img, [points2], [1400, 200, 250])

            blk = np.zeros(img.shape, np.uint8)
            blk[backPosition[2]:backPosition[3], backPosition[0]:backPosition[1]] = img_resize00[:, :, :]
            img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)

        # 旋转比例
        roatBar = np.interp(mm, [20, 340], [600, 250])
        roatPer = np.interp(mm, [20, 340], [0, 100])

        cv2.rectangle(img, (250, 250), (275, 600), (87, 201, 0), 3)
        cv2.rectangle(img, (250, int(roatBar)), (275, 600), (87, 201, 0), cv2.FILLED)
        cv2.putText(img, f'{int(roatPer)} %', (240, 630), cv2.FONT_HERSHEY_COMPLEX, 0.75, (87, 201, 0), 2)
        # print(mm)

        h1, w1, _ = img_model.shape                     # initial img size
        newH = ((h1 + scale) // 2) * 2                  # update img size
        newW = (int((w1 / h1 * newH)) // 2) * 2

        if (cy - newH // 2) > 0 and (cy + newH // 2) < 960 and (h1 + scale) >= 2 and (w1 + scale) >= 2 \
                and (cx - newW // 2) > 0 and (cx + newW // 2) < 1280:  # 边界条件，约束上下左右及缩小范围
            img_model = cv2.resize(img_model, (newW, newH))
            img = cvzone.overlayPNG(img, img_model, [cx - newW // 2, cy - newH // 2])

        else:
            cv2.putText(img, str('It\'s out of range!'), (120, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

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
            if clickt and cursor[0] > 200:
                cv2.circle(img2, cursor, 15, (0, 255, 0), cv2.FILLED)
                x0, y0, w0, h0 = cursor[0], cursor[1], 300, 300
                track_window = (x0 - w0, y0 - h0, w0 - 60, h0 - 60)

                roi = img2[y0 - h0:y0 - 60, x0 - w0:x0 - 60]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])  # 直方图
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

    # UI界面显示
    img = cvzone.overlayPNG(img, img_bbox, [0, 0])              # 边界框
    img = cvzone.overlayPNG(img, imglogo, [15, 13])             # NUDT Logo
    img = cvzone.overlayPNG(img, img_battery, [1140, 15])       # Battery Logo
    img = cvzone.overlayPNG(img, img_compass, [1000, 660])      # Compass Logo

    # FPS显示
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (1142, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 205, 50), 3)

    # 调试状态查看
    # cv2.putText(img, str(backIcon), (1142, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 205, 50), 3)
    # cv2.putText(img, str(backIcon2), (1142, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 205, 50), 3)
    # cv2.putText(img, str(mode), (1142, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 205, 50), 3)

    cv2.imshow('Image', img)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

