import os
import shutil

"""copy normalized images and divide them into classes

Image Name Pattern: First 3 Characters Matches Class Name

input:  normalize_imgs/ (auto)
output: dataset/(classe)
"""

imgs = os.listdir("normalize_imgs")
for img_name in imgs:
    new_path = img_name[0:3]
    shutil.copy2("normalize_imgs/"+img_name, "dataset/"+new_path+"/"+img_name)
