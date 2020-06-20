# CRAFT-custom-use-historical-chinese

clovaai의 craft를 사용하여 고문서에서 글자를 더 잘 찾게 하기 위해 커스텀 하였으며 원본은 해당 링크[https://github.com/clovaai/CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) 에서 확인 할 수 있다.


![이미지1](/image/iaa_d001001b00.jpg)

![이미지2](/image/noname02.png)
![이미지3](/image/noname03.png)


## make_labeling.py

CRAFT에서 얻은 region score를 합성을 통해 각 글자의 labeling을 진행

## cutting.py

make_labeling.py를 통해 얻은 값으로 각 글자별로 자름

## rename_bin02.py

학습을 위한 이진화 이미지로 변경

## img_post.py

그외 필요한 작업물들
