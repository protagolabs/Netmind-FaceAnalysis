import sys
import os
import datetime
import numpy as np
import cv2
import argparse
import face_detection 

os.environ["CUDA_VISIBLE_DEVICES"] = '6'


def main(args):
    _file = "R50-retinaface.onnx"
    detector = face_detection.get_retinaface("_file")

    img = cv2.imread(args.input)
    bbox, _ = detector.detect(img)

    print(bbox) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run MFR online validation.')
    parser.add_argument('--input', type=str, default='demo/0.png',
                        help='input image path')

    args = parser.parse_args()

    main(args)
