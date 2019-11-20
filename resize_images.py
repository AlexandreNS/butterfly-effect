import os
import numpy as np
import cv2 as cv

"""run before: cropped.py
Resizes images to a specified width

input:  width
        mask_out/ (auto)
ouput:  resize_mask_out/
"""
width = 640
imgs = os.listdir("mask_out")
for img_name in imgs:
    img = cv.imread("mask_out/"+img_name, cv.IMREAD_COLOR)
    height = int(img.shape[0] * (width / img.shape[1]))
    dim = (width, height)
    resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    print(img_name ,'Resized Dimensions : ',resized.shape)
    cv.imwrite("resize_mask_out/"+img_name, resized)
