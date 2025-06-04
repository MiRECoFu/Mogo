<p align="center">
  <img src="assets/Mogo_main_00.png" alt="example"/>
</p>
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

<p align="center">
  <img src="assets/main-figure_00.png" alt="example"/>
</p>

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



## MOGO-GUI
We provide a simple GUI for MOGO. You can easily run it by casting

```
python app.py
```

## 🏫About us
Thank you for your interest in this project. We are a startup company, if you are interested in our project, please contact us.

## 🦄 Performance

<p align="center">
  <img src="assets/Mogo_demo_1_00.png" alt="example"/>
</p>

<p align="center">
  <img src="assets/Mogo_demo_2_00.png" alt="example"/>
</p>

### 🏕️ Complex Scenarios
<div align="center">
  <img src="https://github.com/lyuwenyu/RT-DETR/assets/77494834/52743892-68c8-4e53-b782-9f89221739e4" width=500 >
</div>

### 🌋 Difficult Conditions
<div align="center">
  <img src="https://github.com/lyuwenyu/RT-DETR/assets/77494834/213cf795-6da6-4261-8549-11947292d3cb" width=500 >
</div>

## Citation
If you use `MOGO` or `MoSA-VQ` in your work, please use the following BibTeX entries:
```
@article{fu2024mogo,
  title={Mogo: RQ Hierarchical Causal Transformer for High-Quality 3D Human Motion Generation},
  author={Fu, Dongjie},
  journal={arXiv preprint arXiv:2412.07797},
  year={2024}
}
```