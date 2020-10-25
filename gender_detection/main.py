'''
Start file for gender detection
Group no. : 3
Project no. : 3
Author : Sarang Chouguley and Sahil Dandekar
command to run this file : 'python main.py -i test.jpg -m models/model.h5
'''

from classes.genderDetect import GenderDetect
import argparse

# parsing arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image", type = str, help = "path to input image")
ap.add_argument("-m","--model", type = str, help = "path to pretrained model")
ap.add_argument("-d", "--face_detector", type = str, help = "path to pretrained model")
ap.add_argument("-ds", "--dataset", type = str, help = "path to training dataset")
args = vars(ap.parse_args())


if __name__ == "__main__":

    gd = GenderDetect(ds_path = args["dataset"])
    gd.predictGender(model_path = args["model"], image_path = args["input_image"])
