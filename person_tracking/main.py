'''
Description : Main file to run the tracking algorithm 
Author : Sarang Chouguley 
Team no: 3
Project no : 3
Command to run this file: 'python main.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input input/example_02.mp4 --output path_to_output_file'
Type : 'python main.py --help' to see all the arguments
'''

from classes.tracker import PersonTracking
import argparse

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

if __name__ == "__main__":

    pt = PersonTracking(prototxt = args["prototxt"], model = args["model"],ip = args["input"],op = args["output"], confidence = args["confidence"], tracker = args["tracker"], skip_frame = args["skip_frames"])
    pt.run()
