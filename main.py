import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



def place_image(s, e, backi, fori):
    print(fori[0, 0])
    for x in range(fori.shape[0]):
        for y in range(fori.shape[1]):
            # print(fori[x, y])
            if (fori[x, y] != np.array([0, 0, 0])).all():
                backi[s + x, e + y] = fori[x, y]






cap = cv.VideoCapture(0)
fgbg = cv.createBackgroundSubtractorMOG2()
# for i in fgbg:
#     print(i)

img = cv.imread("C:\\Users\\Parsa\OneDrive\\university\\semester 5\\AI\\FinalProject\\drop.png")
img = cv.resize(img, (50, 50))

while (True):
    ret, frame = cap.read()
    bg_learning_rate = 0
    fgmask = fgbg.apply(frame, learningRate=0.2)
    # print(type(fgbg))
    success, film = cap.read()
    # print(type(film))

    # for x in img:
    #     for y in img:
    #         # print(x, y)
    #         film[x, y] = img[x, y]

    place_image(k, 0, film, img)

    cv.imshow('frame00', fgmask)
    cv.imshow('frame', film)
    k = cv.waitKey(1) & 0xff
    if k == 27:
        break
cap.release()
cv.destroyAllWindows()

