import os
from pathlib import Path
import shutil

# ==== 配置路径 ====
img_list_path = r"F:\donkey_face\img_list.txt"                      # 你的图像名列表
txt_label_folder = r"F:\face_detection_results_by_horse_weight_70\face_detection_results_by_horse_weight_70" # 所有txt文件的目录
output_folder = "D:\Data\selected_donkey_face_detection_labels"            # 筛选后txt的输出目录

output_folder.mkdir(exist_ok=True, parents=True)

# ==== 读取图像名 ====
with open(img_list_path, "r") as f:
    img_names = [line.strip() for line in f if line.strip()]

# ==== 遍历并复制对应的 txt 文件 ====
for img_name in img_names:
    stem = Path(img_name).stem  # 去掉.jpg后缀
    label_path = txt_label_folder / f"{stem}.txt"
    if label_path.exists():
        shutil.copy(label_path, output_folder / label_path.name)
    else:
        print(f"❗标签文件不存在: {label_path}")

print("✅ 筛选完毕！")
