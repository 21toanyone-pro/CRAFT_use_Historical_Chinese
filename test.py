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
import natsort

import make_labeling
import cutting
import rename_bin
import rename_bin02
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
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.3, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.28, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.8, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)
image_list = natsort.natsorted(image_list)
result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, image_path, refine_net=None):
    t0 = time.time()
    img_h,img_w,c = image.shape
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    h, w ,c = image.shape
    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    y, feature = net(x)
    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy() #리전 스코어 Region score

    score_link = y[0,:,:,1].cpu().data.numpy() #어피니티 스코어
    # refine link
    if refine_net is not None:
        y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()
    

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, 0.4, poly) # CRAFT에서 박스를 그려주는 부분
    
    
    # # coordinate adjustment #좌표설정
    
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    
    #print(scores)

    for k in range(len(polys)):
         if polys[k] is None: polys[k] = boxes[k]
    t1 = time.time() - t1
    
    # render results (optional)
    render_img = score_text.copy()
    
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)
    Plus_score_text = imgproc.cvMakeScores(render_img) ##

    filename, file_ext = os.path.splitext(os.path.basename(image_path))

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))
    resize_folder = './resize' # resize된 원본 이미지 저장

    if not os.path.isdir(resize_folder+'/'):
        os.makedirs(resize_folder +'/')
    
    resize_file = resize_folder + "/resize_" + filename + '_mask.jpg' #오리지널 이미지



    IMG_RGB2 = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) #craft에서 resize한 이미지를 RGB로 컨버트


    # 합성 이미지를 만들기 위한 부분
    pil_image=Image.fromarray((IMG_RGB2* 255).astype(np.uint8)) 
    images = np.array(pil_image)
    images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(images, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)#+ cv2.THRESH_OTSU
    # 이미지 합성을 위해 이진화

    text_score = cv2.resize(Plus_score_text, None,fx=2, fy=2, interpolation = cv2.INTER_LINEAR) # 다시 원본 사이즈로 조절

    
    thresh = cv2.resize(thresh, (img_w,img_h)) # 원본 이진화 이미지
    text_score = cv2.resize(text_score, (img_w,img_h)) # Region 스코어 이진화 이미지

    text_score=Image.fromarray((text_score).astype(np.uint8))
    text_score = np.array(text_score)

    #thresh=img_post.img_proc(text_score, thresh) # 

    if not os.path.isdir('./og_bri'+'/'): # 원본 이진화 이미지 저장 폴더
        os.makedirs('./og_bri' +'/')
    
    if not os.path.isdir('./score/'): # 스코어 이진화 이미지 저장 폴더
        os.makedirs('./score/')

    cv2.imwrite('./og_bri' + "/og_" + filename + '.jpg', thresh) # 원본 이진화 이미지 저장
    cv2.imwrite('./score' + "/score_" + filename + '.jpg', text_score) # 스코어 이진화 이미지 저장

    img_h = thresh.shape[0]
    img_w = thresh.shape[1]

    IMG_RGB2= cv2.resize(IMG_RGB2, (img_w, img_h)) # 다시 원본 사이즈로 resize
    cv2.imwrite(resize_file, IMG_RGB2)
    
    return boxes, polys, ret_score_text



if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()
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
        image = imgproc.loadImage(image_path)
        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, image_path, refine_net)
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        #cv2.imwrite(mask_file, score_text) #히트맵
        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
    
    print("Labeling...")
    sleep(3)    
    make_labeling.saveLabel() #라벨링 한 것들을 정리 해서 다시 저장
    print("Cropping...")
    sleep(3)
    cutting.cutting_main(img_path_list)
    print("BR...")
    sleep(3)
    rename_bin02.rename()

    print("elapsed time : {}s".format(time.time() - t))





