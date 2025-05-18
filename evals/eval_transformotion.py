import os
from os.path import join as pjoin
import sys

from tqdm import tqdm
# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
import torch

from models.transformers.transformotion import Transformotion
from models.vq.model import RVQVAE

from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper

import utils.eval_t2m as eval_t2m
from utils.fixseed import fixseed

import numpy as np

def load_vq_model(vq_opt):
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

def load_trans_model(model_opt, which_model, vq_model):
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

if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)
    
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263
    
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    out_dir = pjoin(root_dir, 'eval')
    os.makedirs(out_dir, exist_ok=True)

    out_path = pjoin(out_dir, "%s.log"%opt.ext)

    f = open(pjoin(out_path), 'w')

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)
    clip_version = 'ViT-L/14'
    
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)
    
    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim
    print(f"vq_opt{model_opt.vq_name}.code_dim==={vq_opt.code_dim}")
    
    dataset_opt_path = '/root/autodl-tmp/checkpoints/kit/Comp_v6_KLD005/opt.txt' if opt.dataset_name == 'kit' \
        else '/root/autodl-tmp/checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22
    
    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)
    
    for file in os.listdir(model_dir):
        if opt.which_epoch != "all" and opt.which_epoch not in file:
            continue
        print('loading checkpoint {}'.format(file))
        vq_model.eval()
        t2m_transformer = load_trans_model(model_opt, file, vq_model=vq_model)
        t2m_transformer.eval()
        
        t2m_transformer.to(opt.device)
        vq_model.to(opt.device)
        
        fid = []
        div = []
        top1 = []
        top2 = []
        top3 = []
        matching = []
        mm = []
        repeat_time = 20
        for i in tqdm(range(repeat_time)):
            with torch.no_grad():
                best_fid, best_div, Rprecision, best_matching, best_mm = \
                    eval_t2m.evaluation_mask_transformer_test_plus_res(eval_val_loader, vq_model, t2m_transformer,
                                                                       i, eval_wrapper=eval_wrapper, force_mask=False, cal_mm=True)
            msg = f"--> \t Eva. repeat {i} :, FID. {best_fid:.4f}, Diversity. {best_div:.4f}, R_precision. {Rprecision}, matching_score_pred. {best_matching}"
            print(msg)
            fid.append(best_fid)
            div.append(best_div)
            top1.append(Rprecision[0])
            top2.append(Rprecision[1])
            top3.append(Rprecision[2])
            matching.append(best_matching)
            mm.append(best_mm)
            
        fid = np.array(fid)
        div = np.array(div)
        top1 = np.array(top1)
        top2 = np.array(top2)
        top3 = np.array(top3)
        matching = np.array(matching)
        mm = np.array(mm)

        print(f'{file} final result:')
        print(f'{file} final result:', file=f, flush=True)

        msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                    f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
        # logger.info(msg_final)
        print(msg_final)
        print(msg_final, file=f, flush=True)
        
    f.close()


# python evals/eval_transformotion.py --name trm_xl_b38_d1024_20250108 --dataset_name t2m --gpu_id 0 --temperature 1 --gumbel_sample --ext trm_xl_b38_d1024_20250108_eval
        
        
        
        
    
