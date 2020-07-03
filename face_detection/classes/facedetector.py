'''
Team no: 3
Project no: 3
Author: Aayushi Jaiswal
Description: Class containing definition of Face Detection using MTCNN algorithm
'''

import cv2
import mtcnn

####### Face Detection Class #######

class FaceDetection:

    '''
    Constructor
    Arguments: 1. input image
    '''
    def __init__(self, img):
        # initialising some variables
        self.face_detector = mtcnn.MTCNN() # detector type
        self.img = img # input image
        self.conf_t = 0.99 # min confidence required

    '''
     Function to read either the picture or the video from webcam
    '''
    def readInput(self):
        # read the image file if provided or start webcam
        if self.img is None:
            print("[INFO]: No input image provided. Starting webcam...")
            self.video = cv2.VideoCapture(0)
            while True:
                self.frame = self.video.read()
                self.frame = self.frame[1]
                self.detect()
                self.showOutput()
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
            self.video.release()
            cv2.distroyAllWindows()
        else:
            print("[INFO]: Loading image..")
            self.frame = cv2.imread(self.img)

    '''
    Function to run detection
    '''
    def detect(self):

        # check for invalid images
        if self.frame is None:
            print("[INFO]: Invalid image found...")
            return
        
        print("[INFO]: Running detection...")
        # resizing image
        dim = (400, int(self.frame.shape[0] * 400 / float(self.frame.shape[1])))
        self.frame = cv2.resize(self.frame, dim, interpolation=cv2.INTER_AREA)  
        
        # changing img to rgb  
        self.img_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        
        # running mtcnn on image
        self.results = self.face_detector.detect_faces(self.img_rgb)
        print("[INFO]: Results - " + str(self.results))
        
    '''
    Function to show output
    '''
    def showOutput(self):

        # don't show results if invalid image given
        if self.frame is None : 
            return
        # formatting output frame
        for res in self.results:
            x1, y1, width, height = res['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            confidence = res['confidence']
            if confidence < self.conf_t:
                continue
            key_points = res['keypoints'].values()

            cv2.rectangle(self.frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
            cv2.putText(self.frame, f'conf: {confidence:.3f}', (x1, y1), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)

            for point in key_points:
                cv2.circle(self.frame, point, 5, (0, 255, 0), thickness=-1)

        cv2.imshow('output', self.frame)

    '''
    Function to run complete program
    '''
    def run(self):

        if self.img is None:
            self.readInput()
        else:
            # read the input image/webcam video
            self.readInput()
            # detect the faces
            self.detect()
            # show output
            self.showOutput()
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        
            
            
            
    

        

        
