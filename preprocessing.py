import cv2 as cv
import numpy as np
import os
from os import listdir
from utils import unsharp_mask


imagesPath = './image_test/'
images = listdir(imagesPath)
counter = 0
counterN = 0

for img in images:

    orgImg = cv.imread(os.path.join(imagesPath, img))
    # (h, w) = orgImg.shape[:2]

    image = np.copy(orgImg)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # correction of nonuniform illumination
    se = cv.getStructuringElement(cv.MORPH_RECT, (8, 8))
    bg = cv.morphologyEx(gray, cv.MORPH_DILATE, se)
    out_gray = cv.divide(gray, bg, scale=255)

    # otsu
    th1 = cv.threshold(out_gray, 0, 255, cv.THRESH_OTSU)[1]

    # EDGE DETECTION
    blurred = cv.GaussianBlur(th1, (3, 3), 0)
    canny = cv.Canny(blurred, 120, 255, 1)

    kernel = np.ones((5, 5), np.uint8)
    dilation = cv.dilate(canny, kernel, iterations=1)

    # FIND CONTOURS

    cnts = cv.findContours(dilation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

    c = max(cnts, key=cv.contourArea)
    rect = cv.minAreaRect(c)
    box = cv.boxPoints(rect)
    box = np.int0(box)

    cv.drawContours(image, [box], 0, (0, 255, 0), 2)

    # WARPING

    width = int(1066 / 2)
    height = int(1606 / 2)

    input1 = np.float32([box[0], box[1], box[2], box[3]])
    output1 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv.getPerspectiveTransform(input1, output1)

    imgOutput = cv.warpPerspective(orgImg, matrix, (width, height), cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))

    # cv.imshow("result", imgOutput)
    # cv.waitKey(0)

    # RECTANGLE DETECTION 2

    # Properly define to-crop area
    infoArea = imgOutput[50:100, int(width/2):int(4*width/5)].copy()

    infoAreaGray = cv.cvtColor(infoArea, cv.COLOR_BGR2GRAY)
    deblurred1 = unsharp_mask(infoAreaGray)

    scale_percent = 300
    # calculate the 50 percent of original dimensions
    width1 = int(deblurred1.shape[1] * scale_percent / 100)
    height1 = int(deblurred1.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width1, height1)

    name = img.split(".")[0]
    cv.imwrite('./cropped/' + str(name) + '.jpg', cv.resize(deblurred1, dsize))
