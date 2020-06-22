# to run this file : python main.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input input/example_02.mp4 --output output/example_02_output_dlib.avi

# Drive file for person tracking

# importing libraries
import cv2
import numpy as np
import argparse
import imutils
import dlib
from classes.centroidtracker import CentroidTracker
from classes.trackableobject import TrackableObject
from classes.video import FPS

# construct argument parse and parse them
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help=" path to .prototxt file of the detection model")
ap.add_argument("-m", "--model", required=True, help = " path to .caffemodel file of the detection model")
ap.add_argument("-i", "--input", type=str, help="path to optional input video file")
ap.add_argument("-o", "--output", type=str, help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.4, help="minimum probability to filter weak detections")
ap.add_argument("-t","--tracker",type=str,default = "dlib" ,help="tracker type dlib or MOSSE. Defult is dlib")
ap.add_argument("-s", "--skip-frames", type=int, default=30, help="# of skip frames between detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# initialize the video writer
writer = None

# initialize the frame dimentions
W = None
H = None

# set tracker name to dispay on video
trackerName = "Dlib correlation"

# initialize the centroid tracker
ct = CentroidTracker(maxDisappeared =40, maxDistance = 50)

# list to store trackers for each detected person
trackers = []

# dictionary to store trackable objects
trackableObjects = {}

# initialize total number of frames processed, so that detection can be done after n frames
totalFrames = 0
totalDown = 0
totalUp = 0

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# if a input video file was not provided, start the webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
else: # otherwise, grab the reference of the input file
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])

fps = FPS().start() # start the fps throughput

# Start the loop for video frames
while True:
    # grab the next frame and handel if video is from webcam or from video file.
    frame = vs.read()
    frame = frame[1] if args.get("input", False) else frame

    # if a video file is being used and frame = none, it means video has ended
    if args["input"] is not None and frame is None:
        break

    # resize the frame to 500 pixels
    dim = (500, int(frame.shape[0] * 500/float(frame.shape[1])))
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    # rotate the frame by 90 degree
    frame = imutils.rotate_bound(frame, 90)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty then set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # initialilzing writer for writing to disk
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W,H), True)

    # initialize the current status along with the list of
    # bounding box rectangles returned by either (1) Object detector
    # (2) Object Tracker
    status = "Waiting"
    rects = []

    # check to see if object derector should be run
    if totalFrames % args["skip_frames"] == 0:

        # set status to detecting and initialize trackers list
        status = "Detecting"
        trackers = []

        # convert the frame to a blob and run the blob through the network
        # and obtain the detection
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W,H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > args["confidence"]:
                # extract the index of the class label
                idx = int(detections[0, 0, i, 1])

                #if the class label is not a person, ignore it
                if CLASSES[idx] != "person":
                    continue

                # compute the (x,y)-coordinates of the bounding boxes for the object
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                if args["tracker"] == "mosse":
                    # initialize the mosse tracker
                    trackerName = "OpenCV MOSSE"
                    tracker = cv2.TrackerMOSSE_create()
                    rect = tracker.init(frame,(startX, startY, endX, endY))
                else:
                    # initialize dlib correlation tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                    tracker.start_track(rgb, rect)

                # add the trakcer to the list of trackers, so that it can be utilized
                # during skiped frames
                trackers.append(tracker)

    # otherwise, utilizing object trackers rather than object detectors,
    # to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status to tracking
            status = "Tracking"

            # update the tracker and grab the updated position
            if args["tracker"] == "mosse":
                # for openCV tracker
                success, pos = tracker.update(frame)
                # convert the pos into tuple of 4 intergers
                (x,y,w,h) = tuple(map(int,pos))
                startX = x
                startY = y
                endX = x+w
                endY = y+h
            else:
                # for dlib tracker
                tracker.update(rgb)
                pos = tracker.get_position()
                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))
            # draw the bounding boxes on the frame
            cv2.rectangle(frame,(startX,startY),(endX,endY),(0,255,0),2)

    # drawing a horizontal line in the center of the frame, once an object 
    # crosses this line we will determine whether the object is moving 'up' or 'down'
    cv2.line(frame, (0, H//2), (W, H//2), (0,255,255), 2)

    # using the centroid tracker to associate old detected centroids with 
    # newly detected centroids
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for current objectID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it 
        # to determine direction
        else:
            # the difference between the y-coordiante of the *current*
            # centroid and the mean of *previous* centroids will tell
            # in which direction the object is moving (negative for # 'up'
            # and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is -ve(indicating the object is moving up) AND
                # the centroid is above the center line, count the object
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True

                # if the direction is +ve(indicating object is moving down) and the centroid
                # is below the center line, count the object
                if direction > 0 and centroid[1] > H//2:
                    totalDown += 1
                    to.counted = True
                
        # store the trackable object in the dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the 
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4,(0, 255,0), -1)

    # construct a tuple of information to be displayed on the frame
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status),
        ("Tracker", trackerName)
    ]

    # print the info in the frame
    for (i, (k,v)) in enumerate(info):
        text = "{}: {}".format(k,v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0, 255), 2)

    # check to see if the frame should be written to disk
    if writer is not None:
        writer.write(frame)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # close if q is pressed
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far
    # and update the FPS counter
    totalFrames += 1
    fps.update()

# stop the timer and show the FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check if writer pointer should be released
if writer is not None:
    writer.release()

# if a webcam is used, stop the webcam
if not args.get("input", False):
    vs.stop()
# otherwise, release the video file pointer
else:
    vs.release()

# close any open windows
cv2.destroyAllWindows()

