import cv2 as cv
import numpy as np
import module as m
import time

COUNTER = 0
TOTAL_BLINKS = 0
CLOSED_EYES_FRAME = 3
cameraID = 0
videoPath = "Video/Your Eyes Independently_Trim5.mp4"
FRAME_COUNTER = 0
START_TIME = time.time()
FPS = 0

camera = cv.VideoCapture(0)

fourcc = cv.VideoWriter_fourcc(*'XVID')
f = camera.get(cv.CAP_PROP_FPS)
width = camera.get(cv.CAP_PROP_FRAME_WIDTH)
height = camera.get(cv.CAP_PROP_FRAME_HEIGHT)
print(width, height, f)
fileName = videoPath.split('/')[1]
name = fileName.split('.')[0]
print(name)

while True:
    FRAME_COUNTER += 1
    ret, frame = camera.read()
    if not ret:
        break

    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    height, width = grayFrame.shape
    circleCenter = (int(width/2), 50)

    image, face = m.faceDetector(frame, grayFrame)
    if face is not None:
        image, PointList = m.faceLandmakDetector(frame, grayFrame, face, False)

        cv.putText(frame, f'FPS: {round(FPS,1)}', (460, 20), m.fonts, 0.7, m.YELLOW, 2)
        RightEyePoint = PointList[36:42]
        LeftEyePoint = PointList[42:48]
        leftRatio, topMid, bottomMid = m.blinkDetector(LeftEyePoint)
        rightRatio, rTop, rBottom = m.blinkDetector(RightEyePoint)

        blinkRatio = (leftRatio + rightRatio) / 2
        cv.circle(image, circleCenter, int(blinkRatio * 4.3), m.CHOCOLATE, -1)
        cv.circle(image, circleCenter, int(blinkRatio * 3.2), m.CYAN, 2)
        cv.circle(image, circleCenter, int(blinkRatio * 2), m.GREEN, 3)

        if blinkRatio > 4:
            COUNTER += 1
            cv.putText(image, f'Blink', (70, 50), m.fonts, 0.8, m.LIGHT_BLUE, 2)
        else:
            if COUNTER > CLOSED_EYES_FRAME:
                TOTAL_BLINKS += 1
                COUNTER = 0
        cv.putText(image, f'Total Blinks: {TOTAL_BLINKS}', (230, 17), m.fonts, 0.5, m.ORANGE, 2)

        mask, pos, color = m.EyeTracking(frame, grayFrame, RightEyePoint)
        maskleft, leftPos, leftColor = m.EyeTracking(frame, grayFrame, LeftEyePoint)

        cv.line(image, (30, 90), (100, 90), color[0], 30)
        cv.line(image, (25, 50), (135, 50), m.WHITE, 30)
        cv.line(image, (int(width-150), 50), (int(width-45), 50), m.WHITE, 30)
        cv.line(image, (int(width-140), 90), (int(width-60), 90), leftColor[0], 30)

        cv.putText(image, f'{pos}', (35, 95), m.fonts, 0.6, color[1], 2)
        cv.putText(image, f'{leftPos}', (int(width-140), 95), m.fonts, 0.6, leftColor[1], 2)
        cv.putText(image, f'Right Eye', (35, 55), m.fonts, 0.6, color[1], 2)
        cv.putText(image, f'Left Eye', (int(width-145), 55), m.fonts, 0.6, leftColor[1], 2)

        cv.imshow('Frame', image)
    else:
        cv.imshow('Frame', frame)

    SECONDS = time.time() - START_TIME
    FPS = FRAME_COUNTER / SECONDS

    key = cv.waitKey(1)
    if key == ord('q'):
        break

camera.release()
cv.destroyAllWindows()
