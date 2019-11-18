from modules.localbinarypatterns import LocalBinaryPatterns
import os
import cv2 as cv
import numpy as np

lbp = LocalBinaryPatterns(24, 8)

f = open("LBP_files/learn.txt", "w")
f.write("")
f.close()

f = open("LBP_files/test.txt", "w")
f.write("")
f.close()

labels = ["001", "002", "003", "004", "005", "006",
    "007", "008", "009", "010"]

print("Extração LBP de aprendizado iniciada:")
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
print("Extração LBP de aprendizado concluida!!!")
print("------------------------------------------------------------------------------")
print("Extração LBP de teste iniciada:")
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
print("Extração LBP de teste concluida!!!")
