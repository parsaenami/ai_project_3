import random as r
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



def place_image(s, e, backi):
    print(img[0, 0])
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (img[x, y] != np.array([0, 0, 0])).all():
                backi[s + x, e + y] = img[x, y]


def drop(mask, film):
    new_frame = film.copy()
    # place_image(0 + k1, t, film)


    # if k1 > 440:
    #     k1 = 0
    # k1 += 4


cap = cv.VideoCapture(0)
fgbg = cv.createBackgroundSubtractorMOG2()
# for i in fgbg:
#     print(i)

img = cv.imread("C:\\Users\\Parsa\OneDrive\\university\\semester 5\\AI\\FinalProject\\drop.png")
img = cv.resize(img, (25, 25))

k1 = 0

execution = []
while (True):


    ret, frame = cap.read()
    bg_learning_rate = 0
    fgmask = fgbg.apply(frame, learningRate=0.2)
    # print(type(fgbg))
    success, film = cap.read()



    # for d in range(1):
    #     execution.append(r.randrange(film.shape[1] // 2))
        # k1 = 0


    # print(type(film))

    # for x in img:
    #     for y in img:
    #         # print(x, y)
    #         film[x, y] = img[x, y]

    # for t in execution:
    #     drop(t, film, img, k1)



    ###place_image(k1, 0, film, img)

    cv.imshow('frame00', fgmask)
    cv.imshow('frame', film)
    k = cv.waitKey(1) & 0xff
    if k == 27:
        break

    # if k1 > 440:
    #     k1 = 0
    # k1 += 4



cap.release()
cv.destroyAllWindows()

