import sys
import os
import datetime
import numpy as np
import cv2
import argparse
import face_detection
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def main(args):
    _file = "R50-retinaface.onnx"
    detector = face_detection.get_retinaface(_file)
    detector.prepare(use_gpu=True, nms=args.nms, ctx=0)

    img = cv2.imread(args.input)

    img = cv2.resize(img, (224, 224))
    dets, _ = detector.detect(img)

    keeps = detector.nms(dets)

    print(keeps)

    for i in keeps:

        x1 = int(dets[i, 0])
        y1 = int(dets[i, 1])
        x2 = int(dets[i, 2])
        y2 = int(dets[i, 3])      

        crop_img = img[x1:x2,y1:y2]

        cv2.imwrite(args.output.replace(".","_"+str(i)+"."), crop_img)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run MFR online validation.')
    parser.add_argument('--input', type=str, default='demo/0.png',
                        help='input image path')
    parser.add_argument('--output', type=str, default='crop.png',
                        help='output image path')
    parser.add_argument('--nms', type=float, default=0.4,
                        help='threshould to remove the low-quality detect results')
    args = parser.parse_args()

    main(args)
