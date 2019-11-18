import os
import cv2 as cv
import numpy as np

imgs = os.listdir("masks")
for img_name in imgs:
    print(img_name, end='')
    mask = cv.imread("masks/"+img_name)
    gray = cv.cvtColor(mask,cv.COLOR_BGR2GRAY)
    _,thresholded = cv.threshold(gray,0,255,cv.THRESH_OTSU)
    bbox = cv.boundingRect(thresholded)
    x, y, w, h = bbox
    print(bbox)
    img = cv.imread("leedsbutterfly/images/"+img_name)
    foreground_mask = mask[y:y+h, x:x+w]
    foreground_img = img[y:y+h, x:x+w]
    cv.imwrite("out_black_surface_mask/"+img_name, foreground_mask)
    cv.imwrite("out_black_surface/"+img_name, foreground_img)
