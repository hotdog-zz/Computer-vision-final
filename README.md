# Computer-vision-final

本仓库使用mmpretrain在CIFAR-100数据集上训练并测试图像分类模型ResNet152和ViT。

## 基础设置

1. 根据[https://mmdetection.readthedocs.io/en/latest/get_started.html](https://mmpretrain.readthedocs.io/en/latest/get_started.html#installation) 配置虚拟环境，并自行安装tensorboard。
2. 从http://www.cs.toronto.edu/~kriz/cifar.html下载CIFAR-100数据集，并按照https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html解压并存放数据集。

## 训练

在config文件夹中包含了两个模型的的config文件，可直接通过mmpretrain方式进行训练，可参考如下代码，更多细节参考[https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html](https://mmpretrain.readthedocs.io/en/latest/user_guides/train.html)

`
python tools/train.py ${CONFIG_FILE}
`

如果需要开启config中验证集损失函数检测，请将hooks文件夹中文件放入mmdet/engine/hooks内，并取消custom_hooks的注释。

## 测试

将google drive中weight文件夹内权重下载后可直接通过mmpretrain方式进行测试，可参考如下代码，更多细节参考[https://mmdetection.readthedocs.io/en/latest/user_guides/test.html](https://mmpretrain.readthedocs.io/en/latest/user_guides/test.html)

`
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
`
