import os
import numpy as np
import cv2 as cv

imgs = os.listdir("out_black_surface")
segs = os.listdir("out_black_surface_mask")
for id in range(len(imgs)):
    print(id)
    src1 = cv.imread("out_black_surface/"+imgs[id], cv.IMREAD_COLOR)
    src1_mask = cv.imread("out_black_surface_mask/"+segs[id], cv.IMREAD_GRAYSCALE)
    src1_mask = cv.cvtColor(src1_mask, cv.COLOR_GRAY2BGR)

    mask_out = cv.subtract(src1_mask,src1)
    mask_out = cv.subtract(src1_mask,mask_out)

    cv.imwrite("mask_out/"+imgs[id], mask_out)
