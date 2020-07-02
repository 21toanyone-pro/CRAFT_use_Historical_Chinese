# CRAFT-custom-use-historical-chinese

clovaai의 craft를 사용하여 고문서에서 글자를 더 잘 찾게 하기 위해 커스텀 하였으며 원본은 해당 링크[https://github.com/clovaai/CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) 에서 확인 할 수 있다.

The clovaai craft was customized to better find the letters in the Historical documents, and the original can be found at the link [https://github.com/clovaai/CRAFT-pytorch] (https://github.com/clovaai/CRAFT-pytorch)).

# Getting started

## Install dependencis
* PyTorch>=0.4.1
* torchvision>=0.2.1
* opencv-python>=3.4.2
* PIL
* natsort

## 사용법 Used

*test 폴더 구성(Configure test folder)

```
-test
    - 태조
        -1권
            - img01.jpg
            - img02.jpg
            - ...
```

* 실행 코드

```
python test.py --trained_model=craft_mlt_25k.pth --test_folder=.\test
```

```
python test.py --trained_model=[weightfile] --test_folder=[folder path to test images]
```


### 코드 진행 과정

![진행 순서](/image/그림4.jpg)


### Input으로 들어가는 원본 이미지

![원본이미지](/image/iaa_d001001b00.jpg)

* 개별 글자를 검출 할 원본 이미지


### test.py를 통해 생성 되는 데이터

![원본이미지](/image/그림1.png)

* region score를 이미지화 하여 저장, 원본 이미지를 이진화 하여 저장(score, og_bri 폴더)


### make_labeling.py 를 통해 생성되는 데이터

![원본이미지](/image/noname07.png)

* CRAFT를 통해 생성된 Region Score를 이진화 한 이미지와 원본을 이진화 한 이미지를 합성하여 합성 이미지를 생성

* 노이즈를 제거하기 위해 글자기 있는 부분을 crop 하고 crop 된 이미지에서 Connected Component Labeling진행

```
def crop_size(poly): 
    if len(poly)==0:
        return
    poly1 = sort_test(poly)
    a =poly1[0][0]
    poly2 = sorted(poly, key=itemgetter(1))
    b= poly2[0][1]
    poly3 = sorted(poly, key=itemgetter(2), reverse=True)
    c =poly3[0][2]
    poly4 = sorted(poly, key=itemgetter(3), reverse=True)
    d=poly4[0][3]
    area = (a,b,c,d)
    return area

def sort_test(poly):
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
```

* 더 정확한 레이블링을 위해 이진화한 이미지에서 구분선을 제거 하기 위한 코드 

```
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
```

* Connected Component Labeling를 진행하여 얻은 정보를 csv_save폴더에 각 페이지 이름별로 csv 파일로 저장
```
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(addImg,connectivity=8)

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
            #일정 사이즈는 넘기고 진행 

            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            if h > cut_h: continue
            if h < int(cut_h/5): continue
            if w > cut_w: continue
            if w < int(cut_w/2): continue
```
레이블링 된 글자 중 너무 크거나 너무 작은 글자는 제외하고 레이블링


### cutting.py를 통해 생성되는 데이터

![원본이미지](/image/8.png)

* 원본 이미지에서 crop된 개별 문자 이미지 데이터

![생성 되는 csv 파일](/image/저장1.jpg)

* crop한 개별 문자 이미지 데이터에 대한 정보

### rename_bin02.py를 통해 생성되는 데이터

![원본이미지](/image/9.png)

* 학습을 하기 위해 crop한 개별 문자 이미지 데이터를 이진화하여 저장


## Data Link
태조실록(Taejo Silok Image Data):
https://drive.google.com/file/d/19vdIQONJcvCDHPFj2hdZ65R6cnWMHhdd/view?usp=sharing


