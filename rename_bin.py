import numpy as np
import os
import glob
import random
import cv2 as cv
import natsort
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont



def rename():
    bground_path = './crop/'
    bground_list = os.listdir(bground_path)
    count = len(bground_list)
    
    for j in range(0, count):
        dir_path = './rename/' + str(j)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path +'/')

        x = natsort.natsorted(bground_list)

        jpg_list = os.listdir(bground_path+str(x[j]))

        k = natsort.natsorted(jpg_list)
        #csv 파일로 저장
        
        
        for i in range(len(jpg_list)):
            new_filename = dir_path+'/'+str(i)+'.jpg'
            img = cv.imread(bground_path+str(x[j])+'/'+str(k[i]))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2RGB)
            cv.imwrite(new_filename, thresh)

    
if __name__ == '__main__':
    rename()