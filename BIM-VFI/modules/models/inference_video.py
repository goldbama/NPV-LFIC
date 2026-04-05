import glob
import numpy
import os
import cv2
import math
import PIL.Image
import torch
import torch.nn.functional as F
import tqdm
import argparse
from moviepy.video.io.VideoFileClip import VideoFileClip
import sys

from torchvision.utils import save_image
from utils.flowvis import flow2img
from utils.padder import InputPadder


##########################################################

##########################################################
def inference_demo(model, ratio, img0, img1, out_name, save_folder, from_array=False):
    import os
    import cv2
    import torch

    if save_folder is not None and not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    time_range = torch.arange(1, ratio).view(ratio - 1, 1, 1, 1).cuda() / ratio

    # ================== ✅ 关键修改：支持两种输入 ==================
    if not from_array:
        img0_np = cv2.imread(img0)[:, :, ::-1]
        img1_np = cv2.imread(img1)[:, :, ::-1]
    else:
        # 已经是 numpy（BGR），转 RGB
        img0_np = img0[:, :, ::-1]
        img1_np = img1[:, :, ::-1]

    # ================== tensor 转换 ==================
    img0 = (torch.tensor(img0_np.transpose(2, 0, 1).copy()).float() / 255.0).unsqueeze(0).cuda()
    img1 = (torch.tensor(img1_np.transpose(2, 0, 1).copy()).float() / 255.0).unsqueeze(0).cuda()

    _, _, h, w = img0.shape

    if h >= 2160:
        scale_factor = 0.25
        pyr_level = 7
        nr_lvl_skipped = 0
    elif h >= 1080:
        scale_factor = 0.5
        pyr_level = 6
        nr_lvl_skipped = 0
    else:
        scale_factor = 1
        pyr_level = 5
        nr_lvl_skipped = 0

    output_img = None  # ✅ 用于返回

    for i in range(ratio - 1):
        dis0 = torch.ones((1, 1, h, w), device=img0.device) * (i / ratio)
        dis1 = 1 - dis0

        results_dict = model(
            img0=img0,
            img1=img1,
            time_step=time_range[i],
            dis0=dis0,
            dis1=dis1,
            scale_factor=scale_factor,
            ratio=(1 / scale_factor),
            pyr_level=pyr_level,
            nr_lvl_skipped=nr_lvl_skipped
        )

        imgt_pred = torch.clamp(results_dict['imgt_pred'], 0, 1)

        # 转 numpy（BGR）
        output_img = (
            (imgt_pred[0] * 255)
            .byte()
            .cpu()
            .numpy()
            .transpose(1, 2, 0)[:, :, ::-1]
        )

    # ================== ✅ 可选保存 ==================
    if save_folder is not None:
        out_path = os.path.join(save_folder, out_name + '.png')
        cv2.imwrite(out_path, output_img)

    return output_img  # ✅ 返回结果（关键！）


