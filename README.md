<p align="center">
  <img src="assets/Mogo_main_00.png" alt="example"/>
</p>
<h2 align="center">MOGO: Residual Quantized Hierarchical Causal
Transformer for High-Quality and Real-Time 3D
Human Motion Generation</h2>
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
    <a href="https://arxiv.org/pdf/2506.05952" target="_blank">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2304.08069-red">
    </a>
    <a href="mailto:p.fang@soton.ac.uk">
        <img alt="email" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>
</p>
<p align="center">
<strong>MOGO: Residual Quantized Hierarchical Causal
Transformer for High-Quality and Real-Time 3D
Human Motion Generation</strong>
    <br>
    <a href='' target='_blank'>Dongjie Fu*</a>&emsp;
    <a href='' target='_blank'>Tengjiao Sun*</a>&emsp;
    <a href='' target='_blank'>Pengcheng Fang*</a>&emsp;
    <a href='' target='_blank'>Xiaohao Cai</a>&emsp;
    <a href='' target='_blank'>Hansung Kim</a>&emsp;
    <br>
    University of Southampton&emsp;
    MOGO
    <br>
</br>

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
- [x] Code
- [x] App.py
- [x] Inference code of MoSA-VQ and Mogo
- [ ] Project Page
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

## Single Motion Sequence Generation
By using the following command, you can quickly generate an image with **MOGO**.

MOGO Generation
```
CUDA_VISIBLE_DEVICES=0 python gen_t2m.py
```

MoSA-VQ Generation
```
CUDA_VISIBLE_DEVICES=0 python gen_t2m.py
```


## MOGO-GUI
We provide a simple GUI for MOGO. You can easily run it by casting

```
python app.py
```

## 🏫About us
Thank you for your interest in this project. We are a startup team, if you are interested in our project, please contact us.

## 🦄 Performance

### 🏕️ Instruction Completion
<p align="center">
  <img src="assets/Mogo_demo_1_00.png" alt="example"/>
</p>

### 🌋 Complex Scenarios
<p align="center">
  <img src="assets/Mogo_demo_2_00.png" alt="example"/>
</p>


## The following are examples of the generated BVH using MOGO.

<p align="center">
  <img src="assets/backflip.gif" width="300"/>
  <img src="assets/injured-collapse-mogo.gif" width="300"/>
  <img src="assets/skips-circle-mogo-raw.gif" width="300"/>
  <img src="assets/stand-swim-mogo.gif" width="300"/>
</p>

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