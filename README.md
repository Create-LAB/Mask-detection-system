# mask-detetcion
## 项目说明
针对当前疫情情况，各大科技公司开始使用科技手段助力防疫。本项目通过yolov3对人脸口罩数据集进行训练，通过该模型可以识别一个人是否带了口罩，如果识别出此人没有佩戴口罩，则语音播报“请佩戴口罩”，如果此人是佩戴了口罩的，那么系统不会有反应。
## 数据集
用来训练的数据集来源于B站UP主：HamlinZheng，非常感谢该UP主的无私奉献～
## 如何使用该项目
链接：https://pan.baidu.com/s/11z6hmBitSHG4TjilDNFJfQ 
提取码：2zvl

将best.pt放入weights文件夹中，创建一个data文件夹将mask.data和mask.name放进去，配置好项目所需的环境之后，在命令行执行

 python detect.py
 便可看到结果。

--0602update: 新增人脸识别功能，只需要创建 data/faces_lib目录，放入头像图片即可

 ## 如何训练自己的数据集
 想要使用自己的数据集进行重新训练，需要对数据集的格式改成VOC数据集的形式，这个在网上有非常多的教程，我训练的时候参考的是
 https://blog.csdn.net/qq_21578849/article/details/84980298
 大家感兴趣也可以自己训练一下
