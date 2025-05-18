import os
from os.path import join as pjoin
import sys
import random
import string
from tqdm import tqdm
from types import SimpleNamespace
import time
# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
import torch
import torch.nn.functional as F

from models.transformers.transformotion import Transformotion
from models.vq.model import RVQVAE

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt

from utils.fixseed import fixseed
from visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical


from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion

from utils.paramUtil import t2m_kinematic_chain

import numpy as np
clip_version = 'ViT-L/14'

def load_vq_model(vq_opt, opt):
    dim_pose = 251 if opt.dataset_name == 'kit' else 263
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = RVQVAE(vq_opt,
                dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location=opt.device)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

def load_trans_model(model_opt, which_model, vq_model, opt):
    clip_version = 'ViT-L/14'
    transformotion = Transformotion(code_dim=model_opt.code_dim, 
                                    vq_model=vq_model, 
                                    clip_dim=768,
                                    clip_version=clip_version,
                                    opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location=opt.device)
    model_key = 'transformotion'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = transformotion.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading transformotion Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return transformotion

def generate_random_string(length=4):
    # 定义可能的字符集合，包括大写字母、小写字母和数字
    characters = string.ascii_letters + string.digits
    # 使用random.choice随机选择字符，并连接成指定长度的字符串
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

opt_dict = {
    "gpu_id": 0,
    "dataset_name": "t2m",
    "name": 'trm_xl_b38_d1024_20250108',
    "checkpoints_dir": '/root/autodl-tmp/checkpoints',
    "repeat_times": 1
    
}
opt = SimpleNamespace(**opt_dict)

opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
torch.autograd.set_detect_anomaly(True)

dim_pose = 251 if opt.dataset_name == 'kit' else 263

root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
model_dir = pjoin(root_dir, 'model')
result_dir = pjoin('/root/autodl-tmp/generation', generate_random_string())
joints_dir = pjoin(result_dir, 'joints')
animation_dir = pjoin(result_dir, 'animations')
os.makedirs(joints_dir, exist_ok=True)
os.makedirs(animation_dir,exist_ok=True)
out_dir = pjoin(root_dir, 'gen')
os.makedirs(out_dir, exist_ok=True)

# out_path = pjoin(out_dir, "%s.log"%opt.ext)

# f = open(pjoin(out_path), 'w')

model_opt_path = pjoin(root_dir, 'opt.txt')
model_opt = get_opt(model_opt_path, device=opt.device)
clip_version = 'ViT-L/14'

vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
vq_opt = get_opt(vq_opt_path, device=opt.device)
vq_model, vq_opt = load_vq_model(vq_opt, opt)

model_opt.num_tokens = vq_opt.nb_code
model_opt.num_quantizers = vq_opt.num_quantizers
model_opt.code_dim = vq_opt.code_dim

# t2m_transformer = load_trans_model(model_opt, opt, 'net_best_fid.tar')
t2m_transformer = load_trans_model(model_opt, 'best_matching.tar', vq_model=vq_model, opt=opt)



t2m_transformer.eval()
vq_model.eval()

t2m_transformer.to(opt.device)
vq_model.to(opt.device)
mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))

def gen_motions(prompt, length, socket = None):
    # parser = EvalT2MOptions()
    # opt = parser.parse()
    # fixseed(opt.seed)


    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    
    def inv_transform(data):
        return data * std + mean

    prompt_list = []
    length_list = []

    est_length = False
    prompt_list.append(prompt)
    length_list.append(length)

    token_lens = torch.LongTensor(length_list)
    token_lens = token_lens.to(opt.device).long()

    m_length = token_lens
    captions = prompt_list

    sample = 0
    kinematic_chain = t2m_kinematic_chain
    converter = Joint2BVHConvertor()

    for r in range(opt.repeat_times):
        print("-->Repeat %d"%r)
        with torch.no_grad():

            motion_ids = torch.zeros(1, length, 6)
            start_time = time.time()
            pred_motions = t2m_transformer.generate(captions, m_length, motion_ids, temperature=1, socket=socket, inv_transform=inv_transform, recover_from_ric=recover_from_ric, converter=converter)


            pred_motions = pred_motions.detach().cpu().numpy()
            # print(f"pred_motions, {pred_motions}, {pred_motions.shape}")
            # 记录结束时间
            end_time = time.time()

            # 计算总耗时（单位：秒）
            elapsed_time = end_time - start_time

            print(f"整个过程执行时间: {elapsed_time:.2f} 秒")
            data = inv_transform(pred_motions)

        for k, (caption, joint_data)  in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
            animation_path = pjoin(animation_dir, str(k))
            joint_path = pjoin(joints_dir, str(k))
            print(f"out anim =====> {animation_path}")
            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()


            bvh_path = pjoin(animation_path, "sample%d_repeat%d_len%d.bvh" % (k, r, m_length[k]))
            _, joint = converter.convert(joint, filename=bvh_path, iterations=100, foot_ik=False)

            # np.save(pjoin(joint_path, "sample%d_repeat%d_len%d.npy"%(k, r, m_length[k])), joint)

            return bvh_path


if __name__ == '__main__':
    onnx_dir = pjoin(model_opt.checkpoints_dir, 
                    model_opt.dataset_name, 
                    model_opt.name, 
                    'model')
    
    # 确保目录存在
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = pjoin(onnx_dir, "transformotion.onnx")
    print(f"onnx path+++++++++++++++++++++++++++{onnx_path}")
    print(f"print(torch.__version__)=========================={torch.__version__}")
    # parser = EvalT2MOptions()
    # opt = parser.parse()
    # bvh_path, save_path,ik_save_path = gen_motions(opt.text_prompt, opt.motion_length, opt.ext)
    # print(f"{bvh_path}, {save_path},{ik_save_path}")
    
    def export_onnx(model, device):
        # 准备示例输入（根据实际输入维度调整）
        motion_ids = torch.zeros(1, 16, 6).to(device)  # 假设典型长度为64
        captions = ["a person dance"]
        m_length = torch.LongTensor([210]).to(device)
        
        
        # 创建生成函数包装器
        class GeneratorWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, captions, m_length, motion_ids):
                return self.model.generate(captions, m_length, motion_ids, temperature=1.0)

        # torch.onnx.enable_log()
        # 导出ONNX
        torch.onnx.export(
            GeneratorWrapper(model),
            (captions, m_length, motion_ids),
            onnx_path,
            input_names=["captions", "m_length", "motion_ids"],
            output_names=["output"],
            dynamic_axes={
                'motion_ids': {1: 'seq_len'},
                'output': {1: 'seq_len'}
            },
            opset_version=17,
            do_constant_folding=True,
            verbose=False,
            # use_external_data=True,  # 拆分参数到外部文件
            export_params=True,             # 必须为 True
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,  # 启用 CUDA 算子
            training=torch.onnx.TrainingMode.EVAL,
        )
    
    device = torch.device("cuda:0")
    model = t2m_transformer
    export_onnx(model, device)
            







# python gen_t2m.py --gpu_id 0 --ext gen_1 --text_prompt "A person is running on a treadmill."
# gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 -b 0.0.0.0:6006 app:app