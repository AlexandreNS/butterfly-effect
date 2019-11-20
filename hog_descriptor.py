from skimage import feature
import os
import cv2 as cv
import numpy as np

"""Extract HOG features from dataset images

input:  dataset/(classe)/learn
        dataset/(classe)/test
        orientations
        pixels_per_cell
        cells_per_block
        labels
output: HOG_files/learn.txt
        HOG_files/test.txt
"""

orientations = 8
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

labels = ["001", "002", "003", "004", "005", "006",
"007", "008", "009", "010"]

f = open("HOG_files/learn.txt", "w")
f.write("")
f.close()

f = open("HOG_files/test.txt", "w")
f.write("")
f.close()


print("HOG learning extraction started:")
f = open("HOG_files/learn.txt", "a")
for label in labels:
    path = "dataset/"+label+"/learn"
    imgs = os.listdir(path)
    for img_name in imgs:
        print(img_name)
        img = cv.imread(path+"/"+img_name)
        (h, hogImage) = feature.hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
        	cells_per_block=cells_per_block, transform_sqrt=True, block_norm="L1",
        	visualize=True, multichannel=True)
        print(len(h))
        f.write(label+" | [")
        for value_h in h:
            f.write(" "+str(value_h))
        f.write(" ]\n")
f.close()
print("HOG extraction of learning completed !!!")
print("------------------------------------------------------------------------------")
print("Test HOG extraction started:")
f = open("HOG_files/test.txt", "a")
for label in labels:
    path = "dataset/"+label+"/test"
    imgs = os.listdir(path)
    for img_name in imgs:
        print(img_name)
        img = cv.imread(path+"/"+img_name)
        (h, hogImage) = feature.hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
        	cells_per_block=cells_per_block, transform_sqrt=True, block_norm="L1",
        	visualize=True, multichannel=True)
        print(len(h))
        f.write(label+" | [")
        for value_h in h:
            f.write(" "+str(value_h))
        f.write(" ]\n")
f.close()
print("Test HOG extraction completed !!!")
