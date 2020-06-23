'''
Team no: 3
Project no: 3
Author: Sarang Chouguley
Description : Class containing definition of Person Tracking Algorithm
'''

import cv2
import numpy as np
import argparse
import dlib
from .centroidtracker import CentroidTracker
from .trackableobject import TrackableObject
from .video import FPS
######## Person Tracking Class ###############

class PersonTracking:

    '''
    Constructor
    Arguments : 1. .prototxt file path 
                2. .model file path 
                3. path to video file
                4. path for output file
                5. confidence
                6. tracker type
                7. skip frames for detection
    '''
    def __init__(self, prototxt, model, ip, op, confidence, tracker, skip_frame):
        
        # list of classes detected by mobilenet ssd detector
        self.detection_classes = ["background", "aeroplane", "bicycle", "bird", "boat",
	    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	    "sofa", "train", "tvmonitor"]

        # initialising some variables
        self.writer = None # writer pointer
        self.W = None # width of video
        self.H = None # height of video
        self.tracker_name = tracker # set tracker type
        self.t_name = None # name of tracker to be displayed
        self.trackableObjects = {} # dict of available trackedObjects
        self.trackers = [] # list to store trackers for each detected person
        trackableObjects = {} # dictionary to store trackable objects
        self.totalFrames = 0 # initialize total number of frames processed, so that detection can be done after n frames
        self.totalDown = 0 # number of person went up
        self.totalUp = 0 # number of person went down
        self.skip_frame = skip_frame # number of frames to skip for detection
        self.confidence = confidence
        self.ip = ip
        self.op = op

        # initialize the centroid tracker
        self.ct = CentroidTracker(maxDisappeared =40, maxDistance = 50)

        # load the serialized model for detection from disk
        print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(prototxt,model)

    '''
    Function to read the video file 
    '''
    def readVideo(self):
        # read from webcam if no file provided
        if self.ip == None:
            print("[INFO] starting video stream...")
            self.vs = cv2.VideoCapture(0)
        # else read from the file
        else:
            print("[INFO] loading video file...")
            self.vs = cv2.VideoCapture(self.ip)
        

    '''
    Function to loop through each frame of video
    '''
    def loopFrames(self):
        while True:
            # read frame from video
            self.frame = self.vs.read()
            self.frame = self.frame[1] if self.ip is not None else self.frame

            # if a video file is used and frame == none, it means video has ended
            if self.ip is not None and self.frame is None:
                break

            # resize the frame to 500 pixels
            dim = (500, int(self.frame.shape[0] * 500/float(self.frame.shape[1])))
            self.frame = cv2.resize(self.frame, dim, interpolation=cv2.INTER_AREA)
            # self.frame = imutils.rotate_bound(self.frame, 90)
            self.rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            # if the frame dimensions are empty then set them
            if self.H is None or self.W is None:
                (self.H, self.W) = self.frame.shape[:2]
            
            # initialilzing writer for writing to disk
            if self.op is not None and self.writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(self.op, fourcc, 30, (self.W,self.H), True)

            # initialize the current status along with the list of
            # bounding box rectangles returned by either (1) Object detector
            # (2) Object Tracker
            self.status = "Waiting"
            self.rects = []

            # check to see if object detector should be run
            if self.totalFrames % self.skip_frame == 0:
                self.status = "Detecting"
                self.detect() # call detect function
            else: # otherwise, run object tracker
                self.status = "Tracking"
                self.track() # call track function

            # call trackCentroid to track centroids
            self.countPeople()

            # format the output frame
            self.formatFrame()

            # write the frame to disk
            if self.writer is not None:
                writer.write(self.frame)
            
            # close if q is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        
        # check if writer pointer should be released
        if self.writer is not None:
            self.writer.release()

        # if a webcam is used, stop the webcam
        if self.ip is None:
            self.vs.stop()
        # otherwise, release the video file pointer
        else:
            self.vs.release()

        # close any open windows
        cv2.destroyAllWindows()

    '''
    Function to run detection 
    '''
    def detect(self):
        # initialise trackers list as empty each time detector is called
        self.trackers = []
        # convert the frame to a blob and run the blob through the network and obtain the detection.
        blob = cv2.dnn.blobFromImage(self.frame, 0.007843, (self.W,self.H), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            c = detections[0, 0, i, 2] # extract the confidence associated with the prediction
            
            # filter out weak detections
            if c > self.confidence:
                idx = int(detections[0, 0, i, 1]) # extract the index of the class label
                
                # if the class lable is not person, ignore it
                if self.detection_classes[idx] != 'person':
                    continue

                # compute the (x,y)-coordinates of the bounding boxes for the object
                box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                (startX, startY, endX, endY) = box.astype("int")

                # initialize the mosse tracker  
                if self.tracker_name == "mosse":
                    self.t_name = "OpenCV MOSSE"
                    tracker = cv2.TrackerMOSSE_create()
                    rect = tracker.init(self.frame,(startX, startY, endX, endY))
                else: # else initialize dlib correlation tracker
                    self.t_name = "Dlib Correlation"
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                    tracker.start_track(self.rgb, rect)

                # add the trakcer to the list of trackers, so that it can be utilized
                self.trackers.append(tracker)

    '''
    Function to run tracker
    '''
    def track(self):
        # loop over the trackers
        for tracker in self.trackers:
            # update the tracker and grab the updated position
            if self.tracker_name == "mosse": # check tracker type
                # for openCV tracker
                success, pos = tracker.update(self.frame)
                # convert the position into tuple of 4 intergers
                (x,y,w,h) = tuple(map(int,pos))
                startX = x
                startY = y
                endX = x+w
                endY = y+h
            else: # for dlib tracker
                tracker.update(self.rgb)
                pos = tracker.get_position()
                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            self.rects.append((startX, startY, endX, endY))
            # draw the bounding boxes on the frame
            cv2.rectangle(self.frame,(startX,startY),(endX,endY),(0,255,0),2)

    '''
    Function to count people
    '''
    def countPeople(self):
        # using the centroid tracker to associate old detected 
        # centroids with newly detected centroids
        objects = self.ct.update(self.rects)

        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for current objectID
            to = self.trackableObjects.get(objectID, None)

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
                    if direction < 0 and centroid[1] < self.H // 2:
                        self.totalUp += 1
                        to.counted = True

                    # if the direction is +ve(indicating object is moving down) and the centroid
                    # is below the center line, count the object
                    if direction > 0 and centroid[1] > self.H//2:
                        self.totalDown += 1
                        to.counted = True
                    
            # update the trackable objects dictionary
            self.trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the 
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(self.frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(self.frame, (centroid[0], centroid[1]), 4,(0, 255,0), -1)

    '''
    Function to format output frames
    '''
    def formatFrame(self):
        # drawing a horizontal line in the center of the frame, once an object 
        # crosses this line we will determine whether the object is moving 'up' or 'down'
        cv2.line(self.frame, (0, self.H//2), (self.W, self.H//2), (0,255,255), 2)

        # construct a tuple of information to be displayed on the frame
        info = [
            ("Up", self.totalUp),
            ("Down", self.totalDown),
            ("Status", self.status),
            ("Tracker", self.t_name)
        ]

        # print the info in the frame
        for (i, (k,v)) in enumerate(info):
            text = "{}: {}".format(k,v)
            cv2.putText(self.frame, text, (10, self.H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0, 255), 2)
    
        # show the output frame
        cv2.imshow("Frame", self.frame)
 
        # increment the total number of frames processed thus far
        # and update the FPS counter
        self.totalFrames += 1


    
    '''
    Function to execute complete algorithm
    '''
    def run(self):
        self.readVideo()
        self.loopFrames()

