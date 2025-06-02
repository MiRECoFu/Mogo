# Mogo (Motion Generation with One-pass)

## 项目概述
Mogo 是一个专注于特定领域（从代码推测可能与运动生成相关）的项目，运用了深度学习技术，特别是 Transformer 架构，实现了文本到运动的生成任务。项目涵盖了模型训练、评估、数据处理等多个环节，同时提供了相应的工具和脚本。

## 代码结构
以下是项目主要文件和目录的简要说明：
### 主要模块
- `Mogo/common/skeleton.py`：包含获取运动骨架运动学树的函数。
- `Mogo/models/transformers/`：
  - `transformotion.py`：负责文本编码，使用 CLIP 模型对原始文本进行编码。
  - `transformotion_trainer.py`：实现了模型的训练逻辑，包括优化器设置、学习率调整、损失计算等。
  - `transformer_decoder.py`：实现了 RoPE（Rotary Position Embedding）技术，用于处理位置信息。
- `Mogo/utils/`：
  - `motion_process.py`：包含运动处理相关的函数，如获取局部姿态。
  - `eval_t2m.py`：用于评估文本到运动生成模型的性能，计算 FID、多样性、R 精度等指标。
- `Mogo/trainers/`：
  - `train_transformotion.py`：主训练脚本，负责加载数据、模型和优化器，调用训练函数进行模型训练。

### 配置文件
- `Mogo/environment_trm.yml`：用于创建项目所需的 Conda 环境。
- `Mogo/requirements.txt`：列出了项目所需的 Python 依赖包。

## 安装步骤
### 1. 创建 Conda 环境
```bash
conda env create -f Mogo/environment_trm.yml
conda activate transformotion
```

### 2. 安装 Python 依赖
```bash
pip install -r Mogo/requirements.txt
```

## 使用说明
### 训练模型
运行 `train_transformotion.py` 脚本进行模型训练：
```bash
python Mogo/trainers/train_transformotion.py
```
在训练之前，你可能需要根据实际情况修改 `TrainT2MOptions` 中的参数，例如数据集路径、模型保存路径等。

### 评估模型
可以使用 `Mogo/utils/eval_t2m.py` 中的 `evaluation_res_transformer_plus_l1` 函数对训练好的模型进行评估：
```python
from Mogo.utils.eval_t2m import evaluation_res_transformer_plus_l1

# 假设已经定义了 val_loader、vq_model、trans 等变量
fid, diversity, R_precision, matching_score_pred, l1_dist = evaluation_res_transformer_plus_l1(
    val_loader, vq_model, trans, repeat_id=1, eval_wrapper=eval_wrapper, num_joint=opt.joints_num
)
```

## 注意事项
- 确保你的环境中已经安装了 CUDA，并且 PyTorch 版本与 CUDA 版本兼容。
- 在运行代码之前，需要准备好相应的数据集，并将数据集路径配置到 `train_transformotion.py` 中。

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License**.

You are free to use this code for **research and non-commercial purposes**.  
**Commercial use is not permitted** without explicit permission.

[Read full license](http://creativecommons.org/licenses/by-nc/4.0/)

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

Copyright (c) 2024 amberjar

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/

You are free to:
- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

Under the following terms:
- Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- NonCommercial — You may not use the material for commercial purposes.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

## 联系信息
如果你有任何问题或建议，请通过以下方式联系我们：
- 邮箱：[p.fang@soton.ac.uk]
- GitHub Issues：[https://github.com/MiRECoFu/Mogo/issues](https://github.com/MiRECoFu/Mogo/issues)