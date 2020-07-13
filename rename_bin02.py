import numpy as np
import os
import glob
import random
import cv2
import natsort
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import shutil

def rename():
    bground_path = './output/crop/'
    bground_list = os.listdir(bground_path)

    for i in bground_list:
        
        king_path = bground_path + str(i) +'/'
        king_list = os.listdir(king_path)
        king_list = natsort.natsorted(king_list)
        for j in king_list:
            value_path = bground_path + str(i) + '/' + str(j) +'/'
            value_list = os.listdir(value_path)
            value_list = natsort.natsorted(value_list)
            for k in value_list:
                img_path = bground_path + str(i) + '/' + str(j) +'/' + str(k) +'/'
                img_list = os.listdir(img_path)
                img_list = natsort.natsorted(img_list)
                for q in img_list:
                    real_path = bground_path + str(i) + '/' + str(j) +'/' + str(k) +'/' + str(q)
                    save_path = './output/bri/' + str(i) + '/' + str(j) +'/' + str(k) +'/' + str(q)

                    if not os.path.isdir('./output/bri/' + str(i) + '/' + str(j) +'/' + str(k)):
                        os.makedirs('./output/bri/' + str(i) + '/' + str(j) +'/' + str(k) +'/')

                    gray_img = Image.open(real_path)
                    gray_img.resize((130,130))
                    w, h = gray_img.size
                    x = (78-int(w/2))
                    y = (78-int(h/2))
                    images = np.array(gray_img)
                    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
                    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
                    pil_image=Image.fromarray(thresh)

                    example_img = Image.new("L", (156, 156),255)
                    example_img.paste(pil_image, (x, y))
                    example_img.save(save_path)
                
if __name__ == '__main__':
    rename()