"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
from time import sleep
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import re

from PIL import Image

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile
import csv
import rename_bin02
import make_labeling
import cutting
import rename_bin
import img_post

from craft import CRAFT
from collections import OrderedDict

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)


if __name__ == '__main__':

    img_path_list =[]
    append_img_list = img_path_list.append
    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        str_path = str(image_path)
        str_path = str_path.replace('.jpg', '')
        str_path = str_path.replace('test', '')
        str_path = re.sub('[\\\‘|\(\)\[\]\<\>`\'….》]', ' ', str_path)
        str_path = str_path.split()
        img_path_list.append(str_path)

    #print("Labeling...")
    #sleep(3)    
    make_labeling.saveLabel() #라벨링 한 것들을 정리 해서 다시 저장
    print("Cropping...")
    sleep(3)
    #cutting.cutting_main(img_path_list)
    print("BR...")
    sleep(3)
    rename_bin02.rename()





