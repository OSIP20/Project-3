'''
Class for Gender Detection
Group no : 3
Project no : 3
Author: Sarang Chouguley
'''
#-------------------libraries needed-----------------------#
import cv2,os
import numpy as np
import mtcnn
from keras.models import load_model, Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

#--------------------class definition---------------------#
class GenderDetect :

    '''
    Constructor 
    Arguments : 1. Path to dataset
    '''
    def __init__(self, ds_path):
        if ds_path is not None:
            self.dataset_path = ds_path
            self.categories = os.listdir(self.dataset_path) # set the categories for detection
            self.labels = [i for i in range(len(self.categories))] # create a list of lable indexes
            self.lab_dict = dict(zip(categories,self.labels)) # create a dictionary mapping each category to lable index
            print("[INFO] Categories for detection : " + str(self.lab_dict))
        else: 
            print("[INFO] No dataset provided for training...")
        self.img_size = 100

    '''
    Function to initialize model architecture
    '''
    def defineArch(self):
        self.model=Sequential()

        self.model.add(Conv2D(16,(3,3),input_shape=data.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(16,(3,3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(8,(3,3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(64,activation='relu'))
        self.model.add(Dense(32,activation='relu'))
        self.model.add(Dense(16,activation='relu'))

        self.model.add(Dense(2,activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    '''
    Function to train the model
    Arguments : 1. training data
                2. target data
    '''
    def trainModel(self, data, target):

        # split dataset into test, train
        train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.4)

        # fit the model
        checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
        history = self.model.fit(train_data,train_target,epochs=30,callbacks=[checkpoint],validation_split=0.2)

        # save the model
        model.save("new_model.h5")
        print("[INFO] Model saved as 'new_model.h5'...")

    '''
    Function to preprocess training data
    Arguments : 1. Path to haarcascade.xml file
    Returns : data and target np arrays
    '''
    def prepareData(self, face_detector_path):

        # initialise list to store data and target values
        data = []
        target = []

        # initialize face detector
        face_detector = cv2.CascadeClassifier(face_detector_path)

        # find images
        for category in self.categories:
            folder_path=os.path.join(self.dataset_path,category)
            img_names=os.listdir(folder_path)
            
            # read images from dataset
            for img_name in img_names:
                img_path=os.path.join(folder_path,img_name)
                img=cv2.imread(img_path) # read the image 
                try:
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
                    # detect face in image
                    faces=face_detector.detectMultiScale(gray,1.3,5) 
                    
                    for (x,y,w,h) in faces:
                        face_img=gray[y:y+w,x:x+w]
                        break
                    resized=cv2.resize(face_img,(self.img_size,self.img_size))
                    data.append(resized) # store it in data list
                    target.append(self.label_dict[category]) # store respective category in target list
                except Exception as e:
                    print("Error in preparing data...")
                    print("[Error] " + str(e))
                    return (None, None)
        
        # convert data and target list into np arrays
        data = np.array(data)/255.0
        data = np.reshape(data,(data.shape[0], self.img_size, self.img_size,1))
        target = np.array(target)
        # convert target to categorical variable
        target = np_utils.to_categorical(target)

        return(data, target)
               
    '''
    Function to predict gender
    Arguments : 1. Path to model
                2. Path to face detection model (i.e Haarcascade)
                3. Path to test image
    '''
    def predictGender(self, model_path, face_detector_path = None, image_path = None):
        
        # initialize some variables
        self.labels_dict={1:'male',0:'female'}
        self.color_dict={0:(0,0,255),1:(0,255,0)}

        # load the model
        try:
            self.model = load_model(model_path)
            print("[INFO] Model loaded successfully")
            print("[INFO] Mode summary :")
            self.model.summary()
        except Exception as e:
            print("[INFO] Model not loaded successfully...")
            print("[Error] " + str(e))
            return

        # load face detector
        # if no face detector is provided then load mtcnn
        if face_detector_path is None:
            print("[INFO] No face detector provided. Loading mtcnn...")
            self.face_detector = mtcnn.MTCNN()
        else: 
            print("[INFO] Face detector provided. Loading Haarcascade...")
            self.face_detector = cv2.CascadeClassifier(face_detector_path)

        # check for input image or webcam
        if image_path is None: # for webcam
            video = cv2.VideoCapture(0) # capture video from webcam
            while(True): # process each frame
                ret, img = video.read()
                # check if webcam could be loaded
                if not ret:
                    print("[ERROR] Can't load video from webcam...")
                    break

                # for mtcnn
                if face_detector_path is None:
                    img = self.useMtcnn(img)
                # for haarcascade
                else:
                    img = self.useHaarCascade(img)
                cv2.imshow('LIVE',img) # print the output

                # close if 'q' is pressed
                key=cv2.waitKey(1) & 0xFF
                if(key== ord('q')):
                    break
            video.release()
            cv2.destroyAllWindows()

        else: # for input image
            # read the image
            img = cv2.imread(image_path) 
            
            # for mtcnn
            if face_detector_path is None:
                img = self.useMtcnn(img)    
            # for haarcascade
            else:
                img = self.useHaarCascade(img)

            # resize the output
            dim = (400, int(img.shape[0] * 400 / float(img.shape[1])))
            img_small = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)  
            cv2.imshow('Output',img_small) # print the output
            print("[INFO] Output displayed...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("[INFO] Output closed...")

    '''
    Function for mtcnn face detection
    '''
    def useMtcnn(self, img):
        conf = 0.99
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = self.face_detector.detect_faces(frame_rgb)
        for res in results:
            x, y, width, height = res['box']
            x, y = abs(x), abs(y)
            x2, y2 = x + width, y + height

            confidence = res['confidence']
            if confidence < conf:
                continue

            key_points = res['keypoints'].values()
            face_img= gray[x:x2,y:y2]
            resized=cv2.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1))
            result=self.model.predict(reshaped) # predict the result
            print("[INFO] Model result : " + str(result))
            label=np.argmax(result,axis=1)[0] 

            # format the output
            cv2.rectangle(img,(x,y),(x2,y2),self.color_dict[label],2)
            cv2.rectangle(img,(x,y-40),(x2,y),self.color_dict[label],2)
            cv2.putText(img, self.labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2)
        return img

    '''
    Function to use haarcasdade face detector
    '''
    def useHaarCascade(self, img):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=self.face_detector.detectMultiScale(gray,1.3,5)  
        # loop through detections
        for (x,y,w,h) in faces:
            face_img=gray[y:y+w,x:x+w]
            resized=cv2.resize(face_img,(100,100))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,100,100,1))         
            result=self.model.predict(reshaped) # predict the result
            print("[INFO] Model result : " + str(result))
            label=np.argmax(result,axis=1)[0] 

            # format the output
            cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],2)
            cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        return img

    '''
    Function to new create model (if user wishes to)
    '''
    def createModel(self, face_detector_path):
        try:
            data, target = self.prepareData(face_detector_path)
            self.defineArch()
            self.trainModel(data, target)
            print("[INFO] Model trained successfully...")
        except Exception as e:
            print("[INFO] Model could not be trained....")
            print("[Error] " + str(e))


 
