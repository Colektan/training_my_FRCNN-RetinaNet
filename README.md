# 训练 Faster-RCNN & RetinaNet

Faster-RCNN相关的代码参考自：https://github.com/bubbliiiing/faster-rcnn-pytorch/tree/bilibili

RetinaNet相关的代码参考自：https://github.com/yatengLG/Retinanet-Pytorch

## 训练步骤

- 从官网下载 VOC 12 数据集后解压到项目根目录
- 修改 voc_annotation.py里面的annotation_mode=2，运行voc_annotation.py，生成 2012_train.txt 和 2012_val.txt
- 调整 train_frcnn.py / train_retinanet 中的相关参数并运行，开始对应模型的训练

## 测试步骤

- 调整 predict.py 中的相关参数，指定模型参数路径，并运行脚本，进行测试