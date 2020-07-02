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

![이미지6](/image/그림3.jpg)
![이미지7](/image/그림2.jpg)


## New file

* make_labeling.py >= CRAFT에서 얻은 region score를 합성을 통해 각 글자의 labeling을 진행

* cutting.py >= make_labeling.py를 통해 얻은 값으로 각 글자별로 자름

* rename_bin02.py >= 학습을 위한 이진화 이미지로 변경

* img_post.py >= 그외 필요한 작업물들


## Result

![이미지4](/image/noname05.png)

![이미지4](/image/9.png)

## Data Link
태조실록(Taejo Silok Image Data):
https://drive.google.com/file/d/19vdIQONJcvCDHPFj2hdZ65R6cnWMHhdd/view?usp=sharing


