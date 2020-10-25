'''
Description: Main file to run the face detection using MTCNN algorithm
Author: Aayushi Jaiswal, Sarang Chouguley
Team no: 3
Project no: 3
Command to run this file: ' python main.py -i inputImage.jpg '
'''

from classes.facedetector import FaceDetection
import argparse

# parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image", type = str, help = "path to input image")
args = vars(ap.parse_args())


if __name__ == "__main__":

    fd = FaceDetection(img = args["input_image"])
    fd.run()
