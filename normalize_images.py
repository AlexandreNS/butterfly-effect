import os
import cv2 as cv
import numpy as np

"""run before: resize_images.py
fixes images to same height and width

input:  resize_mask_out/ (auto)
output: normalize_imgs/
"""

imgs = os.listdir("resize_mask_out")
alturas = []
largura = 0
for img_name in imgs:
    img = cv.imread("resize_mask_out/"+img_name)
    largura = img.shape[1]
    alturas.append(img.shape[0])
altura_max = max(alturas)

canvas = np.zeros((altura_max, largura, 3))
cv.imwrite("fundo.png", canvas)

imgs = os.listdir("resize_mask_out")
for img_name in imgs:
    print(img_name)
    canvas = cv.imread("fundo.png", cv.IMREAD_COLOR)
    img = cv.imread("resize_mask_out/"+img_name, cv.IMREAD_COLOR)
    canvas[0:img.shape[0], 0:img.shape[1]] = img[0:, 0:]

    cv.imwrite("normalize_imgs/"+img_name, canvas)
