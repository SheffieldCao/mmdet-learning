# Joint Learnning of Instance Segmentation and Depth Estimation
  <img src="https://user-images.githubusercontent.com/12907710/137271636-56ba1cd2-b110-4812-8221-b4c120320aa9.png"/>

[toc]

## Introduction

This repo is based on **OpenMMLab** [mmdet](https://github.com/open-mmlab/mmdetection). 

The master branch works with **PyTorch 1.8.10**.

## Overview of On-shelf Bullets

Results and models are available in the [model zoo](docs/en/model_zoo.md).

<div align="center">
  <b>Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Necks</b>
      </td>
      <td>
        <b>Loss</b>
      </td>
      <td>
        <b>Common</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
      <ul>
        <li>VGG (ICLR'2015)</li>
        <li>ResNet (CVPR'2016)</li>
        <li>ResNeXt (CVPR'2017)</li>
        <li>MobileNetV2 (CVPR'2018)</li>
        <li><a href="configs/hrnet">HRNet (CVPR'2019)</a></li>
        <li><a href="configs/empirical_attention">Generalized Attention (ICCV'2019)</a></li>
        <li><a href="configs/gcnet">GCNet (ICCVW'2019)</a></li>
        <li><a href="configs/res2net">Res2Net (TPAMI'2020)</a></li>
        <li><a href="configs/regnet">RegNet (CVPR'2020)</a></li>
        <li><a href="configs/resnest">ResNeSt (ArXiv'2020)</a></li>
        <li><a href="configs/pvt">PVT (ICCV'2021)</a></li>
        <li><a href="configs/swin">Swin (CVPR'2021)</a></li>
        <li><a href="configs/pvt">PVTv2 (ArXiv'2021)</a></li>
        <li><a href="configs/resnet_strikes_back">ResNet strikes back (ArXiv'2021)</a></li>
        <li><a href="configs/efficientnet">EfficientNet (ArXiv'2021)</a></li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="configs/pafpn">PAFPN (CVPR'2018)</a></li>
        <li><a href="configs/nas_fpn">NAS-FPN (CVPR'2019)</a></li>
        <li><a href="configs/carafe">CARAFE (ICCV'2019)</a></li>
        <li><a href="configs/fpg">FPG (ArXiv'2020)</a></li>
        <li><a href="configs/groie">GRoIE (ICPR'2020)</a></li>
        <li><a href="configs/dyhead">DyHead (CVPR'2021)</a></li>
      </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/ghm">GHM (AAAI'2019)</a></li>
          <li><a href="configs/gfl">Generalized Focal Loss (NeurIPS'2020)</a></li>
          <li><a href="configs/seesaw_loss">Seasaw Loss (CVPR'2021)</a></li>
        </ul>
      </td>
      <td>
        <ul>
          <li><a href="configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py">OHEM (CVPR'2016)</a></li>
          <li><a href="configs/gn">Group Normalization (ECCV'2018)</a></li>
          <li><a href="configs/dcn">DCN (ICCV'2017)</a></li>
          <li><a href="configs/dcnv2">DCNv2 (CVPR'2019)</a></li>
          <li><a href="configs/gn+ws">Weight Standardization (ArXiv'2019)</a></li>
          <li><a href="configs/pisa">Prime Sample Attention (CVPR'2020)</a></li>
          <li><a href="configs/strong_baselines">Strong Baselines (CVPR'2021)</a></li>
          <li><a href="configs/resnet_strikes_back">Resnet strikes back (ArXiv'2021)</a></li>
        </ul>
      </td>
    </tr>
</td>
    </tr>
  </tbody>
</table>

Some other methods are also supported in [projects using MMDetection](./docs/en/projects.md).

## Experiments

### 1. X50 GN & WS

通过引入ResNeXt50，结合 group norm 以及 weight standard。提升效果

### 2. GPWin + SFP `(0421)`

Refer to [Exploring Plain Vision Transformer Backbones for Object Detection](https://arxiv.org/abs/2203.16527).

### 3. OHEM 难样本挖掘 `(0426)`

直接在原有的 (RandomSampler) 基础上设定 OHEM 采样器即可。
```python
model = dict(
  train_cfg=dict(
    rcnn=dict(
      sampler=dict(type='OHEMSampler')
    )
  )
)
```

### 4. $2 \times WinBlock \to WinBlock+SwinBlock$ `(0427)`

### 5. DCNv2 in SFP and GPBlock `(0427)`
DCV v2 的实现配置
```python
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
```
- 考虑在 X50 backbone 中使用 `DCNv2`
- 考虑在 GPWin/GPSwin 中的GPModule使用 `DCNv2`

**TODO RPN activation type -> GELU?**

### N. Results

**All** based on `Mask RCNN`. :sunglasses:

| Backbone |  epochs  |  lr  |w/ COCO pretrain|  **OHEM**  | RPN hidden layers | DCN(X50/GP) | bbox mAP  |  Mask mAP |
|----------|:--------:|:----:|:--------------:|:----------:|:-----------------:|:-----------:|:---------:|:---------:|
| X50+FPN          |  55  | 1e-3 |  Y   |   N  |  1  |  N  |  **0.369**  |  **0.33**   |
| X50+FPN          |  55  | 1e-3 |  N   |   N  |  1  |  N  |  0.363  |  0.315  |
| X50+FPN finetune |  20  | 1e-3 |  Y   | ==Y==|  1  |  N  |  ?  |  ?  |
| X50+FPN finetune |  20  | 1e-3 |  N   | ==Y==|  1  |  N  |  ?  |  ?  |
| X50+FPN finetune |  25  | 1e-4 |  Y   | ==Y==|  1  |  N  |  0.365  |  0.325  |
| X50+FPN finetune |  25  | 1e-4 |  N   | ==Y==|  1  |  N  |  0.358  |  0.306  |
| GPWin+SFP        |  30  | 1e-3 |  N   |   N  |  1  |  N  |  0.215  |  0.177  |
| GPWin+SFP        |  30  | 1e-3 |  N   |   N  |  1  |  N  |  0.215  |  0.177  |
| GPWin+SFP        |  30  | 1e-3 |  N   |   Y  |  2  |  Y  |  ?  |  ?   |
| GPWin+SFP        |  30  | 1e-3 |  N   |   N  |  1  |  N  |  0.215  |  0.177  |
| GPWin+SFP        |  30  | 1e-3 |  N   |   Y  |  2  |  Y  |  ?  |  ?   |

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Installation and Tutorials

Please refer to [get_started.md](docs/en/get_started.md) for installation.

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMDetection.
We provide [detection colab tutorial](demo/MMDet_Tutorial.ipynb) and [instance segmentation colab tutorial](demo/MMDet_InstanceSeg_Tutorial.ipynb), and full guidance for quick run [with existing dataset](docs/en/1_exist_data_model.md) and [with new dataset](docs/en/2_new_data_model.md) for beginners.
There are also tutorials for [finetuning models](docs/en/tutorials/finetune.md), [adding new dataset](docs/en/tutorials/customize_dataset.md), [designing data pipeline](docs/en/tutorials/data_pipeline.md), [customizing models](docs/en/tutorials/customize_models.md), [customizing runtime settings](docs/en/tutorials/customize_runtime.md) and [useful tools](docs/en/useful_tools.md).

Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.

