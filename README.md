## PASS - Official PyTorch Implementation
![](./framework.png)

### [CVPR2021 Oral] Prototype Augmentation and Self-Supervision for Incremental Learning
Fei Zhu, Xu-Yao Zhang, Chuang Wang,  Fei Yin, Cheng-Lin Liu<br>
[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.pdf)
### Usage 
We run the code with torch version: 1.10.0, python version: 3.9.7
* Train CIFAR100
```
python main.py
```
* Train Tiny-ImageNet
```
cd Tiny-ImageNet
python main_tiny.py
```
* Train ImageNet-Subset
```
cd ImageNet-Subset
python main_PASS_imagenet.py
```

### Citation 
```
@InProceedings{Zhu_2021_CVPR,
    author    = {Zhu, Fei and Zhang, Xu-Yao and Wang, Chuang and Yin, Fei and Liu, Cheng-Lin},
    title     = {Prototype Augmentation and Self-Supervision for Incremental Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {5871-5880}
}
```

### Reference
Our implementation references the codes in the following repositories:
* <https://github.com/DRSAD/iCaRL>
* <https://github.com/hankook/SLA>

### Contact
Fei Zhu (zhufei2018@ia.ac.cn)
