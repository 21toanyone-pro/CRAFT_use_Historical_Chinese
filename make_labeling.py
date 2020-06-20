import numpy as np
import cv2
import math
import os
import imgproc
import csv
import natsort
import cutting
import rename_bin
from PIL import Image, ImageFilter
from PIL import ImageFont 
from PIL import ImageDraw 
import img_post
from operator import itemgetter


def saveLabel():
    path = './score/'
    path_list = os.listdir(path)
    count = len(path_list)
    resize_path = './og_bri/'
    resize_list = os.listdir(resize_path)

    res_path = './resize/'
    re_list = os.listdir(res_path)

    csv_root = './csv_save/' #라벨링한 CSV 파일을 저장할 경로
    save_path = './save_rec/' #상자 이미지를 저장할 경로
    
    kk = 0 # 상자가 표시 된 이미지의 장 수

    csv_og= './csv/' # 오리지널 Csv 파일
    txt_list = os.listdir(csv_og) #csv 파일 경로
    cut_h = 0
    cut_w = 0

    x = natsort.natsorted
    for i in range(0, count): 
        C =[]
        poly =[]
        del C[:]
        main =[]
        del main[:]
    
        x = natsort.natsorted(path_list) #라벨링 할 이미지
        y = natsort.natsorted(resize_list) # 상자를 그려 줄 이미지
        z = natsort.natsorted(re_list)

        csv_srot = natsort.natsorted(txt_list) # csv 파일의 csv를 정렬
        csv_f = open(csv_og+csv_srot[i], 'r', encoding='UTF8')
        line_c = csv.reader(csv_f)

        append_C = C.append
        for lines in line_c:
            append_C(lines)
        #===============Crop을 위한 소팅==========================================#

        append_Poly = poly.append
        append_Main = main.append
        for k in range(0,len(C)):
            append_Poly(C[k])
            real =[int(poly[k][0]),int(poly[k][1]),int(poly[k][4]),int(poly[k][5])]
            append_Main(real)
        #=========================================================================#
        area = img_post.crop_size(main)


        img_score = Image.open('./score/'+x[i]) # 스코어 이미지
        img_og = Image.open('./og_bri/'+y[i]) # 원본 이미지
        img_re = Image.open('./resize/'+z[i])

        re_img = img_re.crop(area) 
        rw, rh = re_img.size
        #crop_re_img = crop_re_img.resize((rw, rh))
        re_img.save('./resize_img/' + z[i])


        #원본 이진화 이미지
        crop_re_img = img_og.crop(area) 
        rw, rh = crop_re_img.size
        crop_re_img = np.array(crop_re_img)
        crop_re_img = img_post.del_line_H_CV2(rh, rw, crop_re_img)
        ret, crop_re_img = cv2.threshold(crop_re_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #스코어 이미지
        crop_score_img = img_score.crop(area) 
        rw, rh = crop_score_img.size
        crop_score_img = np.array(crop_score_img)
        ret, crop_score_img = cv2.threshold(crop_score_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        addImg = cv2.add(crop_score_img, crop_re_img)
        cv2.imwrite('./post2/save_'+str(kk)+'.jpg', addImg)

        crop_re_img = cv2.imread('./resize_img/'+z[i], cv2.IMREAD_COLOR)

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(addImg,connectivity=8)

        csv_list = os.listdir('./csv/')
        c = natsort.natsorted(csv_list)

        res_file_csv = csv_root + c[i]
        csvfile = open(res_file_csv,'w',newline='')
        
        csvwriter = csv.writer(csvfile)

        
        ss =0

        if int(rw/15) < cut_w and cut_w != 0:
            cut_w = cut_w
            cut_h = cut_h
        else:
            cut_h = int(rh/18)
            cut_w = int(rw/15)


        for i in range(1, nlabels):
            
            size = stats[i, cv2.CC_STAT_AREA]

            if size < 300: continue
            elif size > 20000: continue
            
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            if h > cut_h: continue
            if h < int(cut_h/5): continue
            if w > cut_w: continue
            if w < int(cut_w/2): continue

            crop_re_img = cv2.rectangle(crop_re_img, (x, y), (x + w, y + h),(0, 0, 255), 1)
            k = x, y, x+w, y+h

            csvwriter.writerow(k) 
            cv2.putText(crop_re_img, str(ss), (x, y), cv2.FONT_ITALIC, 0.2, (0,0,255), 1)
            ss = ss+1
            
        
        cv2.imwrite(save_path+'save_'+str(kk)+'.jpg', crop_re_img)
        print(save_path+'save_'+str(kk)+'.jpg')
        kk =kk+1
    