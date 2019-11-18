# from skimage import exposure
from skimage import feature
import os
import cv2 as cv
import numpy as np

# hog = cv.HOGDescriptor()
# print("teste")
f = open("HOG_files/learn.txt", "w")
f.write("")
f.close()

f = open("HOG_files/test.txt", "w")
f.write("")
f.close()

labels = ["001", "002", "003", "004", "005", "006",
    "007", "008", "009", "010"]

print("Extração HOG de aprendizado iniciada:")
f = open("HOG_files/learn.txt", "a")
for label in labels:
    path = "dataset/"+label+"/learn"
    imgs = os.listdir(path)
    for img_name in imgs:
        print(img_name)
        img = cv.imread(path+"/"+img_name)
        # (h, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
        # 	cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
        # 	visualize=True, multichannel=True)
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        (h, hogImage) = feature.hog(img, orientations=8, pixels_per_cell=(8, 8),
        	cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
        	visualize=True, multichannel=True)
        print(len(h))
        # hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        # hogImage = hogImage.astype("uint8")
        # cv.imshow("HOG Image", hogImage)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        f.write(label+" | [")
        for value_h in h:
            f.write(" "+str(value_h))
        f.write(" ]\n")
f.close()
print("Extração HOG de aprendizado concluida!!!")
print("------------------------------------------------------------------------------")
print("Extração HOG de teste iniciada:")
f = open("HOG_files/test.txt", "a")
for label in labels:
    path = "dataset/"+label+"/test"
    imgs = os.listdir(path)
    for img_name in imgs:
        print(img_name)
        img = cv.imread(path+"/"+img_name)
        # (h, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
        # 	cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
        # 	visualize=True, multichannel=True)
        (h, hogImage) = feature.hog(img, orientations=8, pixels_per_cell=(8, 8),
        	cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
        	visualize=True, multichannel=True)
        print(len(h))
        f.write(label+" | [")
        for value_h in h:
            f.write(" "+str(value_h))
        f.write(" ]\n")
f.close()
print("Extração HOG de teste concluida!!!")
