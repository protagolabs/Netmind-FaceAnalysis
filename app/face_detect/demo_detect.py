import sys
import os
import datetime
import numpy as np
import cv2
import argparse
import face_detection
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '0'



def padscale(img_rgb, x1, y1, x2, y2, scale_factor=0.2):
    

    ## Calculate center points and rectangle side length
    width = x2 - x1
    height = y2 - y1
    cX = x1 + width // 2
    cY = y1 + height // 2
    M = (abs(width) + abs(height)) / 2


    ## Get the resized rectangle points
    newLeft = max(0, int(cX - scale_factor * M))
    newTop = max(0, int(cY - scale_factor * M))
    newRight = min(img_rgb.shape[1], int(cX + scale_factor * M))
    newBottom = min(img_rgb.shape[0], int(cY + scale_factor * M))

    return newLeft, newTop, newRight, newBottom
def main(args):

    img = cv2.imread(args.input)


    _file = "R50-retinaface.onnx"
    detector = face_detection.get_retinaface(_file)
    detector.prepare(use_gpu=True, nms=args.nms, ctx=0)

    img_det = cv2.resize(img, (224, 224))
    dets, lds = detector.detect(img_det)

    keeps = detector.nms(dets)


    for i in keeps:



        # we inverse the coordinate to the orginal images
        x1 = dets[i, 0] / 224 * img.shape[1]
        y1 = dets[i, 1] / 224 * img.shape[0]
        x2 = dets[i, 2] / 224 * img.shape[1]
        y2 = dets[i, 3] / 224 * img.shape[0]     
        
        x1, y1, x2, y2 = padscale(img, x1, y1, x2, y2, scale_factor=args.scale)

        
        

        # #  you may want to 
        crop_img = img[y1:y2,x1:x2]
        print()

        cv2.imwrite(args.output.replace(".","_"+str(i)+"."), crop_img)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run MFR online validation.')
    parser.add_argument('--input', type=str, default='demo/0.png',
                        help='input image path')
    parser.add_argument('--output', type=str, default='crop.png',
                        help='output image path')
    parser.add_argument('--nms', type=float, default=0.4,
                        help='threshould to remove the low-quality detect results')
    parser.add_argument('--scale', type=float, default=0.8,
                        help='padding scale')
    args = parser.parse_args()

    main(args)
