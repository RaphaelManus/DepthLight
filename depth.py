import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import torch

from scripts.depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'./scripts/depth_anything_v2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model.cuda()
model.eval()

def depthany(raw_image, input_size=518):
    depth = model.infer_image(raw_image, input_size)
    return depth