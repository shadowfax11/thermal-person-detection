import cv2
import numpy as np
cv2.namedWindow("preview")
cameraID = 1
vc = cv2.VideoCapture(cameraID)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    # cv2.imshow("preview", frame)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    frame_v = frame_hsv[:,:,2]
    # cv2.imshow("preview", frame_v)
    thresh = 50
    edges = cv2.Canny(frame_v,thresh,thresh*2, L2gradient=True)
    # cv2.imshow("preview", edges)
    blurredBrightness = cv2.bilateralFilter(frame_v,9,150,150)
    thresh = 70
    edges = cv2.Canny(blurredBrightness,thresh,thresh*2, L2gradient=True)
    # cv2.imshow("preview", edges)
    _,mask = cv2.threshold(blurredBrightness,200,1,cv2.THRESH_BINARY)
    erodeSize = 5
    dilateSize = 7
    eroded = cv2.erode(mask, np.ones((erodeSize, erodeSize)))
    mask = cv2.dilate(eroded, np.ones((dilateSize, dilateSize)))
    cv2.imshow("preview", cv2.resize(cv2.cvtColor(mask*edges, cv2.COLOR_GRAY2RGB) | frame, (640, 480), interpolation = cv2.INTER_CUBIC))
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break