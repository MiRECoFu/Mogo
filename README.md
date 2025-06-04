<h2 align="center">Mogo (Motion Generation with One-pass)</h2>
<p align="center">
    <a href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank" rel="noopener noreferrer">
        <img alt="License: CC BY-NC 4.0" src="https://img.shields.io/badge/license-CC--BY--NC%204.0-lightgrey">
    </a>
    <a href="https://github.com/MiRECoFu/Mogo/issues" target="_blank">
        <img alt="GitHub issues" src="https://img.shields.io/github/issues/MiRECoFu/Mogo?color=orange">
    </a>
    <a href="https://github.com/MiRECoFu/Mogo/pulls" target="_blank">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/MiRECoFu/Mogo">
    </a>
    <a href="https://github.com/MiRECoFu/Mogo/stargazers" target="_blank">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/MiRECoFu/Mogo?color=brightgreen">
    </a>
    <a href="https://arxiv.org/pdf/2412.07797" target="_blank">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2304.08069-red">
    </a>
    <a href="mailto:p.fang@soton.ac.uk">
        <img alt="email" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>
</p>

This is the official implementation of papers 
- NeurIPS 2025 under review

---
## 🚀 Updates
- \[2024.06.03\] Reorganize github
- \[2024.05.21\] Submit data process and evaluation algorithms
- \[2024.11.23\] Fix some bugs.
- \[2024.08.04\] Release model architecture.

## 🔥🔥🔥 Todo
- [x] Project Page
- [x] Code
- [x] App.py
- [x] Inference code of MoSA-VQ and Mogo
- [ ] Release pretrained weights train on HumanML3D
- [ ] Release pretrained weights train on our own made huge motion dataset
- [ ] Code of infinite length continuation and generation
- [ ] Controllable motion generation

## Installation

### Conda environment setup
```
conda create -n mogo python=3.10 -y
conda activate mogo
pip install -r requirement.txt
```

## Single Image Generation
By using the following command, you can quickly generate an image with **MIGC**.
```
CUDA_VISIBLE_DEVICES=0 python inference_single_image.py
```
The following is an example of the generated image based on stable diffusion v1.4.
 
<p align="center">
  <img src="figures/MIGC_SD14_out.png" alt="example" width="200" height="200"/>
  <img src="figures/MIGC_SD14_out_anno.png" alt="example_annotation" width="200" height="200"/>
</p>

By using the following command, you can quickly generate an image with **MIGC++**, where both the box and mask are used to control the instance location.
```
CUDA_VISIBLE_DEVICES=0 python migc_plus_inference_single_image.py
```
The following are examples of the generated images using MIGC++.

<p align="center">
  <img src="figures/migc++_output.png" alt="example" width="1000" height="300"/>
</p>

## MIGC-GUI
We have combined MIGC and [GLIGEN-GUI](https://github.com/mut-ex/gligen-gui) to make art creation more convenient for users. 🔔This GUI is still being optimized. If you have any questions or suggestions, please contact me at zdw1999@zju.edu.cn.

![Demo1](videos/video1.gif)


## 🏫About us
Thank you for your interest in this project. We are a startup company, if you are interested in our project, please contact us.

## 🦄 Performance

### 🏕️ Complex Scenarios
<div align="center">
  <img src="https://github.com/lyuwenyu/RT-DETR/assets/77494834/52743892-68c8-4e53-b782-9f89221739e4" width=500 >
</div>

### 🌋 Difficult Conditions
<div align="center">
  <img src="https://github.com/lyuwenyu/RT-DETR/assets/77494834/213cf795-6da6-4261-8549-11947292d3cb" width=500 >
</div>

## Citation
If you use `Mogo` or `MoSA-VQ` in your work, please use the following BibTeX entries:
```
@article{fu2024mogo,
  title={Mogo: RQ Hierarchical Causal Transformer for High-Quality 3D Human Motion Generation},
  author={Fu, Dongjie},
  journal={arXiv preprint arXiv:2412.07797},
  year={2024}
}
```