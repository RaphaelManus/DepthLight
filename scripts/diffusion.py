from share import *
import os
import sys

scripts_directory = os.path.abspath("./scripts/Panodiff")
sys.path.append(scripts_directory)

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import json
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
import numpy as np
import cv2
from PIL import Image
#from scripts.Panodiff.cldm.logger import ImageLogger ################ REPLACED ################
from scripts.Panodiff.cldm.model import create_model, load_state_dict
import torch
import gc


panodiff_model_path = './scripts/panodiff/models/norota_inpaint.yaml'
test=False

num_training = 500
exclude_360 = True
batch_size = 1

# Rotation Supervision
rotation_supervise = False
rotation_loss_lambda = 5e-4
roll_augment = True
roll_schedule = False
padding_augment = True


logger_freq = 1 #num_training // batch_size * 2
learning_rate = 1e-5
resume_path = './scripts/panodiff/pretrained_models/norota_clean.ckpt'

NUM_GPUS = 1
N_acc = 2 # gradient accumulate
max_epochs = 20 * N_acc
sd_locked = True
only_mid_control = False
img_size = 512
log_path = 'logs'
name_head = 'public'

class SingleDataset_(VisionDataset):
    def __init__(self, root, dataset_name, index,
                prompt,
                prompt_path=None, extensions='.jpg', 
                Train=True, down_scale=1, 
                break_iter=None, mask=None):
        
        self.height = 512
        self.width = 1024
        transform, target_transform = self.init_crops_transform()
        
        super(SingleDataset_, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        
        if break_iter is None:
            self.break_iter = 0
        else:
            self.break_iter = break_iter + 1
        self.extensions = extensions
        self.train = Train
        self.dataset_type = dataset_name
        prompt_json_file_train = prompt_path
        self.prompts = json.load(open(prompt_json_file_train, 'r'))

        self.mask = np.expand_dims(mask, axis=-1)
        
        # reading the full pano
        target = root
        print(root.shape)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        source = target * self.mask

        self.target = (target.astype(np.float32) / 127.5) - 1.0
        self.hint = source.astype(np.float32) / 255.0
        self.prompt = prompt

    def __getitem__(self, index):
        return dict(jpg=self.target, 
                    txt=self.prompt, 
                    hint=self.hint, 
                    mask=np.where(self.mask==0, 1, 0))

    def __len__(self):
        return 5 # generate 5 instances for the same prompt
        
    def init_crops_transform(self):
        transform = transforms.Compose(
                                  [transforms.Resize((int(self.height), int(self.height))), transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                   ])
        target_transform=transforms.Compose(
                                  [transforms.Resize((int(self.height), int(self.height))), transforms.ToTensor(),
                                   transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                   ])
        
        return transform, target_transform
    
    
    def normal_transform(self, image):
        image_arr = np.array(image)
        image_arr = (image_arr / 255).astype(np.float32)
        return torch.tensor(image_arr)
    
    def loader(self, path, down_scale=1):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if down_scale > 1:
            width, height = img.size
            new_width = int(width / down_scale)
            new_height = int(height / down_scale)
            new_size = (new_width, new_height)
            img = img.resize(new_size)
        return img

def get_model():
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(panodiff_model_path) #.cuda()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=True) # cuda:0
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    model.rotation_supervise = rotation_supervise
    model.use_gt_rots = True
    model.padding_augment = padding_augment
    if rotation_supervise:
        model.rotation_loss_lambda = rotation_loss_lambda
    model.roll_augment = roll_augment
    model.roll_schedule = roll_schedule
    model.deform_augment = False
    model.eval()
    return model

def panodiff(target, mask, prompt):
    test_dataset = SingleDataset_(root=target, 
                                    prompt_path='./scripts/panodiff/prompts/my_sun360_prompts_no360.json',
                                    dataset_name='sun360',
                                    index='000000',
                                    Train=False,
                                    prompt=prompt,
                                    extensions='.png',
                                    mask=mask
                                    )

    return DataLoader(test_dataset, 
                    num_workers=1, 
                    batch_size=batch_size, 
                    shuffle=False)

@torch.no_grad()
def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                use_ema_scope=True,
                **kwargs):
    use_ddim = ddim_steps is not None

    log = dict()
    
    z, c = self.get_input(batch, self.first_stage_key, bs=N) # takes 5go of vram #bs=N

    c_cat, c_control, c = c["c_concat"][0][:N], c["control_input"][0][:N], c["c_crossattn"][0][:N]
    N = min(z.shape[0], N)
    n_row = min(z.shape[0], n_row)
    log["reconstruction"] = self.decode_first_stage(z)

    log["control"] = c_control[:, -3:] * 2.0 - 1.0  # NOTE(wjh): only output rgb layers.
    _mask = c_control[:, :1]
    #log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

    if sample:
        # get denoise row
        samples, z_denoise_row = self.sample_log(
            cond={"c_concat": [c_cat], "control_input": [c_control], "c_crossattn": [c]},
            batch_size=N, ddim=use_ddim,
            ddim_steps=ddim_steps, eta=ddim_eta, mask=_mask)
        x_samples = self.decode_first_stage(samples)
        log["samples"] = x_samples
        if plot_denoise_rows:
            denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
            log["denoise_row"] = denoise_grid

    if unconditional_guidance_scale > 1.0:
        # NOTE(wjh):
        # used in condition guided sample:
        # model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        uc_cross = self.get_unconditional_conditioning(N)
        uc_control = c_control  # torch.zeros_like(c_cat)
        uc_full = {"control_input": [uc_control], "c_crossattn": [uc_cross], "c_concat": [c_cat]}
        # uc_full = {"c_crossattn": [uc_cross], "c_concat": [c_cat]}
        gc.collect()
        torch.cuda.empty_cache()
        # TODO: find the difference with the original causing more vram usage
        samples_cfg, _ = self.sample_log(
            # cond={"c_concat": [c_cat], "c_crossattn": [c]},
            cond={"c_concat": [c_cat], "c_crossattn": [c], "control_input": [c_control]},
            batch_size=N, ddim=use_ddim,
            ddim_steps=ddim_steps, eta=ddim_eta,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=uc_full,
            # mask=c_cat[:, :1],
            # x0=z[:N]
        )
        # trick, make the ends meet
        if self.padding_augment:
            pad_length = samples_cfg.shape[-1] // 8
            samples_cfg = self.rolling_padding(samples_cfg, pad_length)
        
        x_samples_cfg = self.decode_first_stage(samples_cfg)
        if self.padding_augment:
            pad_length_hr = pad_length * 8
            x_samples_cfg = x_samples_cfg[..., pad_length_hr:-pad_length_hr]
        # x_samples_cfg = torch.clip(x_samples_cfg, min=-1, max=1)
        log[
            f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg  # x_samples_cfg *  _mask + log["control"] * (1 - _mask)
        # log["pred_x0"] = self.decode_first_stage(_['pred_x0'][0])

    return log
