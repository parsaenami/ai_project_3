import numpy as np
import cv2 as cv

from matplotlib import pyplot as plt

cap = cv.VideoCapture(0)
fgbg = cv.createBackgroundSubtractorMOG2()


while (1):
    ret, frame = cap.read()
    bg_learning_rate = 0
    fgmask = fgbg.apply(frame)
    # print(type(fgbg))
    cv.imshow('frame', fgmask)
    k = cv.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()
