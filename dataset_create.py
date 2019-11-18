import os
import shutil

imgs = os.listdir("normalize_imgs")
for img_name in imgs:
    new_path = img_name[0:3]
    shutil.copy2("normalize_imgs/"+img_name, "dataset/"+new_path+"/"+img_name)
