import imutils
from imutils import face_utils
import cv2 
import numpy as np 
import dlib 

print("[INFO] loading dlib thermal face detector...")
detector = dlib.simple_object_detector('./thermal-facial-landmarks-detection/models/dlib_face_detector.svm')

cv2.namedWindow("preview")
cameraID = 1
vc = cv2.VideoCapture(cameraID)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    frame_v = frame_hsv[:,:,2]
    thresh = 50
    edges = cv2.Canny(frame_v,thresh,thresh*2, L2gradient=True)
    blurredBrightness = cv2.bilateralFilter(frame_v,9,150,150)
    thresh = 70
    edges = cv2.Canny(blurredBrightness,thresh,thresh*2, L2gradient=True)
    _,mask = cv2.threshold(blurredBrightness,200,1,cv2.THRESH_BINARY)
    erodeSize = 5
    dilateSize = 7
    eroded = cv2.erode(mask, np.ones((erodeSize, erodeSize)))
    mask = cv2.dilate(eroded, np.ones((dilateSize, dilateSize)))
    
    image_copy = frame.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the image 
    rects = detector(frame_gray, upsample_num_times=1)
    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    image_copy = cv2.resize(cv2.cvtColor(mask*edges, cv2.COLOR_GRAY2RGB) | image_copy, (640, 480), interpolation = cv2.INTER_CUBIC)
    cv2.imshow("preview", image_copy)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27:
        break
