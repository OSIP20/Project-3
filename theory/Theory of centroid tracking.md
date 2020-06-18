# This file tries to explain the theory behind the working of Centroid Tracking Algorithm

- This algorithm is not developed by us, we have simply used it in our project. You can find link to the original blog containing the algorithm, in the reference section

 - The implementation of this algorithm can be found inside the Centroid Object Trackging folder under the code folder

-------------------------------------------------------------------

## The algorithm -

1. Read frame from video stream.
2. Pass the frame thorugh a object detector and get the coordiantes of the detected objects (e.g. Face or Person)
3. Draw rectangle around the detected objects.
4. Maintain a list of detected objects in each frame, update the list for each frame.
5. Read the next frame and repeate above steps till the video stream ends.

- In the above algorithm, we are using a object detection model to perfom object tracking.
- Any object detection model can be used as long as it is fast and accurate. As a slow model cannot be used for real time tracking.
- The heart of this algorithm is centroid tracker class, this class calculates centroid of each detected object and maintains a list of such centroids in each frame.
- The centroids are updated, registered or deregistered in each frame.

**Case 1: Updating an object's centroid.**

- This is done by calculating euclidean distance between each pair of centroids present in 2 consecutive frames.
- The pair having the minimum distance between them is most likely of the same object.
- So the centroid of the object is updated with the new centroid.

**Case 2: Registering a new object**

- If the number of centroids in the current frame is more than the number of centroids in the previous frame, it means a new object has been detected.
- So the centroids having maximum distance from the already detected centroids are new centroids.
- These centroids are added in the detected objects list. (i.e. Registered).

**Case 3: Deregistering a object's centroid**

- A object is deregistered if has been marked as disappeared in n consecutive frames (n can be decided by you).
- If the number of centroids in current frames is less than the number of centroids in the previous frame, it means the some objects have disappeared in the current frame.
- So the centroids having maximum distance from the registered centroids, are the disappeared centroids.
- These centroids are marked as disappeared.

## Problem with this algorithm -

- As object detection is applied in each frame, this makes the algorithm computationaly expensive.

## Solution to this problem -

- A combination of object tracking and object detection models can be used.
- Object detection can be applied after every nth frame, and Object tracking can be applied when object is not being detected.
- This ensures accuracy as well as efficiency of the algorithm.

-------------------------------------------------------------------

## References

- Simple object tracking with OpenCV by PyImageSearch [https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/]
