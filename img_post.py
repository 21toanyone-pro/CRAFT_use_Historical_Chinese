import sys
import os
import time
import argparse
from time import sleep

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile


import glob
import re
import csv
import pandas as pd
import natsort
from operator import itemgetter

def crop_size(poly): #글자 부분인 곳을 자르기 위한 코드
    if len(poly)==0:
        return
    poly1 = sort_test(poly) # 중간에 비어있는 부분을 보완하기 위한 코드
    a =poly1[0][0]
    poly2 = sorted(poly, key=itemgetter(1))
    b= poly2[0][1]
    poly3 = sorted(poly, key=itemgetter(2), reverse=True)
    c =poly3[0][2]
    poly4 = sorted(poly, key=itemgetter(3), reverse=True)
    d=poly4[0][3]
    area = (a,b,c,d)
    return area

def sort_test(poly): #중간에 비어있는 이미지를 보완하기 위한 코드
    a =[]
    poly = sorted(poly, key=itemgetter(1))
    if len(poly) <= 13:
        for i in range(0, len(poly)):
            a.append(poly[i])
    else:
        for i in range(0, len(poly)):
            a.append(poly[i])
    a = sorted(a, key=itemgetter(0))
    return a


def add_img(score_img, bl_img): # 이미지 합성
    addImg = cv2.add(score_img, bl_img)
    #addImg = cv2.resize(addImg, (1664, 2560))
    return addImg

def del_line_H_CV3(height,weight,thresh): # 이진화 한 이미지에서 구분선을 지우기 위한 코드
    BW_CHECK = 0
    for W in range(weight-1):
        BW_CHECK = 0
        for H in range(height-1):
            px = thresh[H, W]
            px_1 = thresh[H+1, W]
            if px>0 and px_1>0:
                BW_CHECK =BW_CHECK + 1
                if BW_CHECK >=230:
                    thresh[0:height,W] = 0
                    thresh[0:height,W-1] = 0
                    thresh[0:height,W-2] = 0
                    thresh[0:height,W+1] = 0
                    break
            elif px > 0 and px_1 ==0:
                BW_CHECK = 0
    return thresh
    
def del_line_H_CV2(height,weight,thresh):# 이진화 한 이미지에서 구분선을 지우기 위한 코드
    BW_CHECK = 0

    for H in range(height-1):
        BW_CHECK = 0
        for W in range(weight-1):
            px = thresh[H, W]
            px_1 = thresh[H, W+1]
            if px>0 and px_1>0:
                BW_CHECK =BW_CHECK + 1
                if BW_CHECK >=200:
                    thresh[H,0:W] = 0
                    break
            elif px > 0 and px_1 ==0:
                BW_CHECK = 0

    for W in range(weight-1):
        BW_CHECK = 0
        for H in range(height-1):
            px = thresh[H, W]
            px_1 = thresh[H+1, W]
            if px>0 and px_1>0:
                BW_CHECK =BW_CHECK + 1
                if BW_CHECK >=230:
                    thresh[0:height,W] = 0
                    thresh[0:height,W-1] = 0
                    thresh[0:height,W-2] = 0
                    thresh[0:height,W+1] = 0
                    break
            elif px > 0 and px_1 ==0:
                BW_CHECK = 0
    return thresh