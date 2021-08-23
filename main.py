from paddleocr import PaddleOCR, draw_ocr
import cv2 as cv
import imutils
import numpy as np
import os


def signaturesStatus(frame):

    # signtures' status
    sg1 = 0
    sg2 = 0

    # load stuff
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    yolo_width_height = (416, 416)

    # don't resize
    frame_resize_width = 480
    confidence_threshold = 0.4
    overlapping_threshold = 0.1

    if frame_resize_width:
        frame = imutils.resize(frame, width=frame_resize_width)
    (H, W) = frame.shape[:2]

    # construct blob of frames by standardization, resizing, and swapping Red and Blue channels (RBG to RGB)
    blob = cv.dnn.blobFromImage(frame, 1 / 255.0, yolo_width_height, swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []

    for laterOutput in layerOutputs:

        for detection in laterOutput:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidence_threshold:
                # scale the bboxes back to the original image size
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX)
                y = int(centerY)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))

    # remove overlapping bounding boxes
    bboxes = cv.dnn.NMSBoxes(
        boxes, confidences, confidence_threshold, overlapping_threshold)

    idx = 0
    coordsX = []

    # get objects' properties
    if len(bboxes) > 0:
        # 'i' is the index in bboxes of boxes of interest (the rest is doo-doo)
        for i in bboxes.flatten():

            # centroid
            (x, y) = (boxes[i][0], boxes[i][1])
            coordsX.append(x)
            # (w, h) = (boxes[i][2], boxes[i][3])

            idx = idx + 1

    # if no signatures are found then clear sg's
    else:
        sg1 = 0
        sg2 = 0

    # if one signature is found, determine side position
    if idx == 1:
        # if signature is closer to left
        if np.abs(coordsX[0]) < np.abs(W - coordsX[0]):
            sg1 = 1
            sg2 = 0
        else:
            sg1 = 0
            sg2 = 1

    # if two signatures are found then set sg's
    elif idx == 2:
        sg1 = 1
        sg2 = 1

    return sg1, sg2


def getDate(frame):

    dd = 0
    mm = 0

    infoAreaGray1 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    bg1 = cv.morphologyEx(infoAreaGray1, cv.MORPH_DILATE, se)
    ready1 = cv.divide(infoAreaGray1, bg1, scale=255)
    th1 = cv.threshold(ready1, 0, 255, cv.THRESH_OTSU)[1]
    result1 = ocr.ocr(th1)

    for line in result1:
        date = line[1][0]
        if date.split('.|/|-| ')[0].isnumeric():
            print(line)
            dd = date.split('.|/|-| ')[0]

            try:
                second = date.split('.|/|-| ')[1]
                mm = second[0:2]
            except:
                mm = 0

    yy = 2021

    return dd, mm, yy

###################################################


with open("./model/classes.names", "r", encoding="utf-8") as f:
    labels = f.read().strip().split("\n")

yolo_config_path = "./model/yolov4-tiny_test.cfg"
yolo_weights_path = "./model/yolov4-tiny_train_best.weights"

useCuda = True
ocr = PaddleOCR(lang="en")

net = cv.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)

if useCuda:
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

###################################################

if __name__ == '__main__':

    image_dirs = ['./image_test']

    # GENERATE TXT FILE

    # truncate existing file first (overwrite?)
    output = open('output.txt', 'w')
    # txt file row array
    rows = []

    # OBTAIN DATA

    total_size = 0

    for image_dir in image_dirs:
        path, dirs, files = next(os.walk(image_dir))

        # loops through each file within './images_test'
        for f in os.listdir(image_dir):
            # make sure selected content is indeed an image
            if f.split(".")[-1] == "jpg":

                # retrieve image
                image = cv.imread(image_dir+'/'+f)

                # retrieve cropped image
                cropped = cv.imread(image_dir+'/'+f.split(".")[0]+'.png')

                # get image id
                idn = f.split(".")[0]

                # get signatures' status
                sign1, sign2 = signaturesStatus(image)

                # get date
                date_day, date_month, date_year = getDate(cropped)

                # create row object
                rows.append([idn, sign1, sign2, date_day, date_month, date_year])

                # acts as an index as well
                total_size += 1

    # WRITE DATA ONTO TXT FILE

    print(f'total images: {total_size}')

    for row in rows:
        output.write(','.join(map(str, row)) + '\n')
