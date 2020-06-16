# This is the driver program for simple object tracking with the help of CentroidTracker 
# USAGE : python mainProgram.py --prototxt deploy.prototxt --model faceDetection.caffemodel

# import the necessary packages
from centroid_class.Tracker import CentroidTracker
import imutils
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help = "Path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to caffe 'pre-trained' model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help = "minimum probalility to filter weak detections")
args = vars(ap.parse_args())

# initialize centroid tracker and frame dimentions
ct = CentroidTracker()
(H, W) = (None, None)

# load the serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream from webcam
print("[INFO] starting video stream")
vs = cv2.VideoCapture(0)


# loop over the frames from the video stream
while True:
    # read the next frame from the video stream
    ret, frame = vs.read()
    frame = imutils.resize(frame, width = 400)

    # if the frame dimensions are none, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the frame, pass it through the network,
    # obtain our output predictions, initialize the list of bounding box rectangles
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # filter out weak detections by ensuring the predictions probability is greater than 
        # minimum threshold
        if detections[0,0,i,2] > args["confidence"] :
            # compute the (x,y) coordinated of the detected object, then update the rects list
            box = detections[0,0,i,3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))

            # draw a bounding box around the object
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # update the centroid tracker using detected box rectangles
    objects = ct.update(rects)

    #loop over the tracked objects 
    for(objectID, centroid) in objects.items():
        # draw both ID and centroid of the detected object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # press 'q' to quit
    if key == ord("q"):
        break

# destroy all pointers and close all windows
vs.release()
cv2.destroyAllWindows()

