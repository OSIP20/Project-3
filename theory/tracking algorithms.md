 # This file discusses about different tracking algorithms available in OpenCV
-------------------------------------------------------------------
#### Following algorithms are available in OpenCV 3.4.1 for object tracking
**1. Boosting**
**2. MIL(Multiple Instance Learning)**
**3. TLD(Tracking, Learning and Detection)**
**4. Medianflow** 
**5. Mosse**
**6. Goturn**
**7. KCF (Kernelized Correlation Filter)**
**8. CSRT**

### Boosting :
- This algo tracks ok even at high frame rate of 60 or 70.
- It does not report when it has lost track of the object.
**Pros**: 
-- None. (Because this algo is very old).
**Cons**:
-- Tracking performance is mediocre.
-- It does not reliably know when tracking has failed.

### MIL:
- This algo is slower but more accurate than Boosting.
- It does not report when it has lost track of the object.
**Pros**:
-- Performance is pretty good.
-- Does not drift as much as the Boosting algo.
-- it does reasonable job under partial occlusion.
**Cons**:
-- Tracking failure is not reported reliably.
-- Does not recover from full occlusion.

### KCF:
- Combines best of both the Boosting and MIL.
- It is fast as well as accurate.
- It reports when it has lost track of the object very accurately.
- It can be used for general purpose.
**Pros**:
-- Accuracy and speed are both better than MIL.
-- Reports tracking failure better than MIL and Boosting.
**Cons**: 
-- Does not recover from full occlusion.
-- not implemented in OpenCV 3.0.

### Median flow:
- Median flow works when the object is moving slowly and predectively.
- It reports when it looses track of the object very accurately.
- this can be used in case of tracking a face in a webcam video.
**Pros**:
-- Excellent tracking failure reporting.
-- Works well when the motion is predictable and there is no occlusion.
**Cons**:
-- Fails under large motion.


### TLD:
- This algo surpasses all other algorithms when there is occlusion(some other object overlaps or comes in front of the object to be tracked).
**Pros**:
-- Works best under occlusion over multiple frames.
-- Tracks best over scale changes.
**Cons**:
-- Lots of false positives making it almost unusable.

### Goturn :
- This is the only algo based on CNN.
- Its robust to viewpoint changes, lighting changes, and deformations.

### CSRT:
- Works well on non-rectangular regions or objects.
- Operates at a comparatively lower fps.
- But gives higher accuracy for object tracking.

-------------------------------------------------------------------
## References - 
- https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
