# File to handel all the video related processing
import datetime
# A Class to calulate fps
class FPS:
    def __init__(self):
        # store the start time, end time and number of frames that were examined between start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()
        return self

    def update(self):
        # increment the total number of frames examined during the start stop interval
        self._numFrames += 1
        return self

    def elapsed(self):
        # return the total number of seconds between the start and end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # return the approx calculated fps
        return self._numFrames / self.elapsed()
