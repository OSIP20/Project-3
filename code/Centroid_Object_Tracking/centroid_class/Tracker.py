####### This is a python class for implementing Centroid Tracker ########
#-----------------------------------------------------------------------#
# for theory and algorithm of this class, please refer to theory folder #

# importing required libraries
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDiasppeared = 40):
        
        # initialize the next unique object id for detected objects
        self.nextObjectID = 0

        # initialize ordered dictionary of detected objects
        self.detectedObjects = OrderedDict()
        
        # initialize ordered dictionary of disappeared objects
        self.disappearedObjects = OrderedDict()

        # store the maximum number of frames a object is allowed to be 
        # marked as disappeared, before being deleted from the detectedObjects dictionary
        self.maxDisappeared = maxDiasppeared

    # register function to register new objects into the detectedObjects dictionary
    def register(self, centroid):
        # when registering an object we use the nextObjectID to 
        # store the id of new detected object
        self.detectedObjects[self.nextObjectID] = centroid

        # setting the disappeared count of the object to 0, as it is a newly detected object
        self.disappearedObjects[self.nextObjectID] = 0
        
        # incrementing the object id variable
        self.nextObjectID +=1

    # function to deregister a object, which has been marked as disappeared 
    # for more than maxDisappeared times
    def deregister(self,centroid):
        # to deregister an object, simply delete it from 
        # detectedObjects and disappearedObjects list
        del self.detectedObjects[self.nextObjectID]
        del self.disappeared[self.nextObjectID]

    # function to update the centroid of detected objects
    def update(self, rects):

        # if rectangle list is empty, that means no new object is detected
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them as disappeared,
            # bcoz they have not been detected in this frame, as rectangle list is empty
            for objectID in list(self.disappearedObjects.keys()):
                self.disappearedObjects[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frame where a given object has been marked as
                # missing, deregister it 
                if self.disappearedObjects[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            
        # return early as there are no centroids or tracking info to update
        return self.detectedObjects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype = "int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if objects have been detected for the first time, register each of them
        if len(self.detectedOjects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids)

        # otherwise, we are currently tracking objects so we need to
        # try to match the input centroids to existing object centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.detectedObjects.key())
            objectCentroids = list(self.detectedObjects.value())

            # compute the distance between each pair of object centroids 
            # and input centroids, respectively 
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # find the row and column index 
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            # 2 variable to store the rows/columns already processed
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):

                # if the rows/columns have already been examined, ignore them
                if row in usedRows or col in usedCols:
                    continue
                else:
                    # otherwise, grab the object ID and change its centroid to new centroid
                    # and reset disappearedObjects counter
                    objectID = objectID[row]
                    self.detectedObjects[objectID] = inputCentroids[col]
                    self.disappearedObjects[objectID] = 0

                    # add these rows, columns to used rows/columns set
                    usedRows.add(row)
                    usedCols.add(col)

            # compute both the row and columns not processed
            unusedRows = set(range(0,D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # if rows > columns in distance array, it means there are more input centorids
            # than already registered centorids, so mark extra input centroids as disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unsed rows 
                for row in unusedRows:
                    # mark such objects as disappeared
                    objectID = objectID[row]
                    self.disappearedObjects[objectID] += 1

                    # check if the object has been disappeared for a long time
                    if self.disappearedObjects[objectID] > self.maxDisappeared:
                        # if yes deregister it
                        self.deregister(objectID)
            
            # else if the number of columns > number of rows in distance array, it means there are new centroids
            # just register them
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
            
            # return the set of trackable/updated objects
            return self.detectedObjects
