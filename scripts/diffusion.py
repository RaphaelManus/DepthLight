from share import *
import os
import sys

scripts_directory = os.path.abspath("./scripts/Panodiff")
sys.path.append(scripts_directory)

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from scripts.Panodiff.dataloaders.rota_dataset import SingleDataset
#from scripts.Panodiff.cldm.logger import ImageLogger ################ REPLACED ################
from scripts.Panodiff.cldm.model import create_model, load_state_dict


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
resume_path = './scripts/panodiff/norota_clean.ckpt'

NUM_GPUS = 1
N_acc = 2 # gradient accumulate
max_epochs = 20 * N_acc
sd_locked = True
only_mid_control = False
img_size = 512
log_path = 'logs'
name_head = 'public'

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
    test_dataset = SingleDataset(root=target, 
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
