from PIL import Image, ImageFilter
from PIL import ImageFont 
from PIL import ImageDraw 

import cv2
import numpy as np
import os
import glob
import re
import csv
import pandas as pd
import natsort
from operator import itemgetter


def sorting_arry(arry):
    for i in range(len(arry)):
        if i+1 >= len(arry):
            return arry
        if abs(int(arry[i][0]) - int(arry[i+1][0])) < 30: #x와 x+1을 뺐을때 35보다 작으면 같은 라인에 있는 거니까 
            arry[i+1][0] = int(arry[i][0])

def same_check(arry, og_arry): # 같은 라인인지 체크하고 원래 글자로 수정해줌
    for i in range(0,len(arry)):
        for j in range(0,len(arry)):
            if arry[i][1] == og_arry[j][1]:
                if arry[i][2] == og_arry[j][2]:
                    if arry[i][3] == og_arry[j][3]:
                        arry[i][0] = og_arry[j][0]
    return arry


# 분류를 위한 왕 코드
links = ['iaa','iba','ica','ida','idb','idc','idd','ide','iea'
,'ifa','iga','iha','iia','ija'
,'ika','ila','ima','ina','inb'
,'ioa','iob','ipa','iqa','ira'
,'irb','isa','isb','ita','itb'
,'iua','iva','iwa','ixa','iya'
,'iza','izb','izc']


## 권 분류를 하기 리스트 생성
l = []
for i in range(1, 10):
    s = 'd00'+str(i)
    l.append(s)
for q in range(10, 100):
    s = 'd0'+str(q)
    l.append(s)
for w in range(100, 300):
    s = 'd'+str(w)
    l.append(s)

m2 =[]

def cutting_main(img_list):
    file_path = './output/crop/' #save_crop_img_path
    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    img_path = './output/resize_img/' # load_img_path
    if not os.path.isdir(img_path):
        os.makedirs(img_path)
    csv_save_path = './output/csv_save/'

    if not os.path.isdir(csv_save_path):
        os.makedirs(csv_save_path)

    bground_list = os.listdir(img_path) #원본 이미지 경로
    txt_list = os.listdir(csv_save_path) #csv 파일 경로
    count = len(bground_list) #이미지 경로 안의 이미지 갯수
    print('Page:'+str(count))
    line_c =[]  #csv 저장
    main =[] #좌표 저장되 있는 리스트
    og_main =[]

    for j in range(0,count): #이건 한 장 마다 한바퀴 돈다. 
        C = []
        point =[]
        del C[:]
        main =[]
        del main[:]
        x = natsort.natsorted(bground_list)
        y = natsort.natsorted(txt_list)

        #==============파일 이름을 거르기 위한 부분===============#
        str_path = str(x[j])
        str_path = str_path.replace('.jpg', '')
        str_path = str_path.replace('mask', '')
        str_path = str_path.replace('resize', '')
        for i in range(0, len(links)):
            str_path = str_path.replace(links[i], '')
        for i in range(0, len(l)):
            str_path = str_path.replace(l[i], '')
        str_path = re.sub('_', ' ', str_path)
        str_path = re.sub(' ', '', str_path)
        #==============파일 이름을 거르기 위한 부분===============#

        img = Image.open(img_path+x[j]) #이미지 오픈 
        
        csv_f = open(csv_save_path+y[j], 'r', encoding='UTF8')
        line_c = csv.reader(csv_f)
        
        for lines in line_c:
            C.append(lines)

        print('글자 수 :' +str(len(C)))
        if len(C) < 5:
            continue
        
        appendP = point.append
        appendMain = main.append
        append_OG = og_main.append
        
        for k in range(0,len(C)):
            appendP(C[k])
            real =[int(point[k][0]),int(point[k][1]),int(point[k][2]),int(point[k][3])]
            real2 =[int(point[k][0]),int(point[k][1]),int(point[k][2]),int(point[k][3])] 
            appendMain(real)
            append_OG(real2)
        #====================================================================================#
        #우측에서 왼쪽으로 세로 읽기로 정렬
        
        main = sorted(main, key=itemgetter(0)) #배열 소팅, reverse=True
        main = sorting_arry(main) #소팅
        main = sorted(main, key=itemgetter(0,1))
        main = sorted(main, key=itemgetter(0), reverse=True)
        main = same_check(main, og_main) #소팅된걸 체크해서 바꿔 줌
        cutting_img_num = 0

        #====================================================================================#
        #배열을 커팅 하는 부분
        king_name = "ㅇ"
        for k in range(len(C)):
            arry1 = int(main[k][0])
            arry2 = int(main[k][1])
            arry3 = int(main[k][2])
            arry4 = int(main[k][3])
            #========파일 생성==========================#
            
            king_path = file_path +str(img_list[j][0])
            book_path = file_path +str(img_list[j][0])+'/'+str(img_list[j][1])+'/'
            dir_path = file_path +str(img_list[j][0])+'/'+str(img_list[j][1])+'/'+str(str_path) # 파일 저장 경로

            if not os.path.isdir(king_path):
                os.mkdir(king_path +'/')  
            if not os.path.isdir(book_path):
                os.mkdir(book_path +'/')  
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path +'/')  

            #========판다스 파일 생성==========================#
            replaceA = txt_list[j].replace('.csv', '')
            location = (arry1,arry2)
            a = abs(arry1 -arry3) #판다스 저장용
            b = abs(arry2 - arry4) #판다스 저장용
            #========파일 생성==========================#
            if arry1 < 0:
                arry1 = 0
            s_name = int(cutting_img_num)
            area =(arry1, arry2, arry3, arry4)
            cropped_img = img.crop(area) # 이미지 크롭

            img_save_path = dir_path+'/'+str(s_name) + '.jpg'

            cropped_img.save(img_save_path) #이미지 저장
            cutting_img_num = cutting_img_num +1
            get_size = os.path.getsize(img_save_path)
            volume_kb = '%0.2f' % (get_size/1024)

            if str(img_list[j][0]) != king_name:
                m2 =[]
            
            m2.append([img_save_path,replaceA, location,a,b,volume_kb+'KB'])

            if not os.path.isdir('./output/'+ str(img_list[j][0])):
                os.mkdir('./output/'+ str(img_list[j][0]) +'/')  
            df2 = pd.DataFrame(m2, columns=['save_path','filename','Location', 'w','h','Volume(KB)'])
            df2.to_csv('./output/'+ str(img_list[j][0]) +'/'+str(img_list[j][0]) +'_data.csv',encoding='euc-kr')

            king_name = str(img_list[j][0])

    
