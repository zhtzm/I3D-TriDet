# 挑战性课程——基于Transformer架构的视频片段查找与分类

## Installation

Refer to [TriDet](https://github.com/dingfengshi/TriDet "[CVPR2023] TriDet: Temporal Action Detection with Relative Boundary Modeling")

1. Please ensure that you have installed PyTorch and CUDA. <br>
  **(This code  We use pyhton=3.8 pytorch version=2.2.1 CDUA=11.8)** Refer to [Pytorch](https://pytorch.org/get-started/locally/)
  ```shell
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

2. Install the required packages by running the following command:

```shell
pip install -r requirements.txt
```

3. Install NMS

```shell
cd ./TriDet/libs/utils
python setup.py install --user
cd ../..
```

4. Done! We are ready to get start!

