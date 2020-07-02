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

test 폴더 구성
Configure test folder

```
-test
    - 태조
        -1권
            - img01.jpg
            - img02.jpg
            - ...
```

```
python test.py --trained_model=craft_mlt_25k.pth --test_folder=.\test
```

```
python test.py --trained_model=[weightfile] --test_folder=[folder path to test images]
```


## 설명

### 진행 순서

![진행 순서](/image/그림4.jpg)


### Input으로 들어가는 원본 이미지

![원본이미지](/image/iaa_d001001b00.jpg)

* 개별 글자를 검출 할 원본 이미지


### test.py를 통해 생성 되는 합성 이미지

![원본이미지](/image/그림1.png)

* CRAFT를 통해 생성된 Region Score를 이진화 한 이미지와 원본을 이진화 한 이미지를 합성하여 합성 이미지를 생성

* region score의 정보를 csv 폴더에 각 페이지 이름별로 csv 파일로 저장

### make_labeling.py 를 통해 생성되는 데이터

![원본이미지](/image/noname07.png)

* 노이즈를 제거하기 위해 글자기 있는 부분을 crop 하고 crop 된 이미지에서 Connected Component Labeling진행

* Connected Component Labeling를 진행하여 얻은 정보를 csv_save폴더에 각 페이지 이름별로 csv 파일로 저장



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


