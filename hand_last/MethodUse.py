import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

def distance(pos1,pos2):
    distance = ((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)/100
    return distance
def fingerStill(handpos,Storedata,count=17):#position of finger,array to store the positon,count means the inerval
    clik = False
    Storedata.append(handpos)
    if len(Storedata) == count :
        if distance(Storedata[-1],Storedata[0])<5:
            Storedata.clear()
            clik = True
        else:Storedata.clear()
    else:
        pass
    return Storedata,clik

def inArea(pos,Leftup,Rightdown):
    if Leftup[0] <= pos[0] <= Rightdown[0] and Leftup[1] <= pos[1] <= Rightdown[1]:
        return True
    else:return False


def cv2ImgAddText(img, text, left, top, textColor=(248, 248, 255), textSize=22):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    # fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    fontStyle = ImageFont.truetype("model_data/PingFang-Heavy-2.ttf", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def readtxt(location,rowCharacter):
    f = open(location, encoding='utf-8')
    txt = []
    txt_line = []
    count = 0
    for line in f:
        count = 0
        letterC = ''
        for letter in line.strip():
            count += 1
            if count <= rowCharacter:
                letterC = letterC + letter
            else:
                letterC = letterC + letter+'\n'
                count = 0
        txt.append(letterC.strip())
    for strT in txt:
        t = strT.split()
        txt_line.append(t)
    return txt_line

