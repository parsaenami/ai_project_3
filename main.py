import random as r
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

index = []

def place_image(start, end, frame):
    print(img[0, 0])
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (img[x, y] != np.array([0, 0, 0])).all():
                frame[start + x, end + y] = img[x, y]


def drop(mask, film):
    new_frame = film.copy()
    for d in range(3):
        index.append([0, r.randrange(new_frame.shape[1] - img.shape[1])])
    for t in index:
        if t[0] >= (mask.shape[0] - 30):
            index.remove(t)
    for i in range(len(index)):
        s, x, e, y = index[i][0], img.shape[0], index[i][1], img.shape[1]
        if (mask[s - x + 6, e - y + 4] == np.array([0, 0, 0])).all():
            if s <= mask.shape[0] - 30:
                index[i] = [s + 30, y]
            place_image(s - x, e - y, new_frame)
        else:
            place_image(s - x, e - y, new_frame)
    return new_frame

# cap = cv.VideoCapture("C:\\Users\\Parsa\\PycharmProjects\\ai_project_3\\1.mp4")
cap = cv.VideoCapture(0)
fgbg = cv.createBackgroundSubtractorKNN(history=5000)

img = cv.imread("C:\\Users\\Parsa\OneDrive\\university\\semester 5\\AI\\FinalProject\\drop.png")
img = cv.resize(img, (7, 7))

k1 = 0

execution = []
while (True):


    ret, frame = cap.read()
    fgmask = fgbg.apply(frame, learningRate=0.02)
    morph = fgmask.copy()
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (4, 2))
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kernel)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel)
    _, morph = cv.threshold(morph, 127, 255, cv.THRESH_BINARY)

    f = drop(morph, frame)

    cv.imshow('frame00', morph)
    cv.imshow('frame', f)
    k = cv.waitKey(1) & 0xff
    if k == 27:
        break



cap.release()
cv.destroyAllWindows()

