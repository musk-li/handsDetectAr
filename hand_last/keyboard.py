import cv2
from PIL import Image
import cvzone
import math
from numpy import *
from cvzone.HandTrackingModule1 import HandDetector
myEquation = ''
delayCounter = 0
windoww = 2280
windowh = 900
cap1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1.set(3, windoww)
cap1.set(4, windowh)
detector=HandDetector(detectionCon=0.8)

class Button():
    def __init__(self, pos, text, size=[70, 70]):
        self.pos = pos
        self.text = text
        self.size = size

    def draw(self, img, color=(0, 0, 0)):
        x, y = self.pos
        w, h = self.size
        cv2.rectangle(img, self.pos, (x + w, y + h), color, cv2.FILLED)
        cv2.putText(img, self.text, (x + 20, y + 35), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1)
        return img
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", 'P', "[", ']'],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";", "'","---"],
        ["Z", "X", "C", "V", "B", "N", 'M', ",", ".", "/","Backspace"]]
buttonList = []
interdistance = 80
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        if i == 2 and j == 10:
            buttonList.append(Button([interdistance * j + 200 + i * 12, 300 + interdistance * i], key,size=[150, 70]))
        elif i == 1 and j == 11:
            buttonList.append(Button([interdistance * j + 200 + i * 12, 300 + interdistance * i], key, size=[100, 70]))
        else:
            buttonList.append(Button([interdistance * j + 200+ i*12, 300+interdistance * i], key))

def AllDraw(img):
    for btn in buttonList:
        img = btn.draw(img)
    return img
while True:
    success, img = cap1.read()
    img = cv2.flip(img, 1)
    allHands, img = detector.findHands(img, draw=True, flipType=True)
    img = AllDraw(img)
    if len(allHands) != 0:
        lmList = allHands[0]['lmList']
        length, _, img = detector.findDistance(lmList[4], lmList[5], img)
        for i , button in enumerate(buttonList):
            x, y = button.pos
            w, h = button.size
            deviation=1
            if x - deviation < lmList[8][0] < x + w + deviation \
                    and y - deviation < lmList[8][1] < y + h + deviation:
                img = button.draw(img, (100, 150, 0))
                if length < 50 and delayCounter == 0:
                    Clickvalue = keys[int(i//12)][int(i%12)]
                    if Clickvalue == 'Backspace' and myEquation != '':
                        myEquation = myEquation[:len(myEquation)-1]
                    elif Clickvalue == '---':
                        myEquation += ' '
                    else:
                        myEquation += Clickvalue
                    delayCounter = 1
        if delayCounter != 0:
            delayCounter += 1
            if delayCounter > 10:
                delayCounter = 0
        blk = np.zeros(img.shape, np.uint8)
        blk[backPosition[2]:backPosition[3], backPosition[0]:backPosition[1]] = img_resize00[:, :, :]
        img = cv2.addWeighted(img, 1.0, blk, Rate1, 1)
    cv2.rectangle(img, (400,100), (800,250),(255, 255, 255), cv2.FILLED)
    cv2.putText(img, 'Message:', (400, 130), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 3)
    cv2.putText(img, myEquation, (400, 170), cv2.FONT_HERSHEY_PLAIN,2, (100, 200, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
