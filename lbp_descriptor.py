from modules.localbinarypatterns import LocalBinaryPatterns
import os
import cv2 as cv
import numpy as np

"""Extract LBP features from dataset images

input:  dataset/(classe)/learn
        dataset/(classe)/test
        labels
        numPoints
        radius
output: LBP_files/learn.txt
        LBP_files/test.txt
"""

numPoints = 24
radius = 8

labels = ["001", "002", "003", "004", "005", "006",
"007", "008", "009", "010"]

lbp = LocalBinaryPatterns(numPoints, radius)

f = open("LBP_files/learn.txt", "w")
f.write("")
f.close()

f = open("LBP_files/test.txt", "w")
f.write("")
f.close()


print("LBP learning extraction started:")
f = open("LBP_files/learn.txt", "a")
for label in labels:
    path = "dataset/"+label+"/learn"
    imgs = os.listdir(path)
    for img_name in imgs:
        print(img_name)
        img = cv.imread(path+"/"+img_name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        hist = lbp.describe(gray)
        # print(hist)
        f.write(label+" | [")
        for value_h in hist:
            f.write(" "+str(value_h))
        f.write(" ]\n")
f.close()
print("LBP extraction of learning completed !!!")
print("------------------------------------------------------------------------------")
print("Test LBP extraction started:")
f = open("LBP_files/test.txt", "a")
for label in labels:
    path = "dataset/"+label+"/test"
    imgs = os.listdir(path)
    for img_name in imgs:
        print(img_name)
        img = cv.imread(path+"/"+img_name)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        hist = lbp.describe(gray)
        # print(hist)
        f.write(label+" | [")
        for value_h in hist:
            f.write(" "+str(value_h))
        f.write(" ]\n")
f.close()
print("Test LBP extraction completed !!!")
