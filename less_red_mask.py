import os
import numpy as np
import cv2 as cv

dataset_path = "leedsbutterfly"
imgs = os.listdir(dataset_path+"/images")
segs = os.listdir(dataset_path+"/segmentations")
for id in range(len(segs)):
    print(id)
    src1_mask = cv.imread(dataset_path+"/segmentations/"+segs[id], cv.IMREAD_GRAYSCALE)
    for i in range(src1_mask.shape[0]):
        for j in range(src1_mask.shape[1]):
            if src1_mask[i, j] != 255 and src1_mask[i, j] != 0:
                src1_mask[i, j] = 0
    cv.imwrite("masks/"+imgs[id], src1_mask)
print("Done!!!")
