'''
The file is a copy of the original script.py used to load another predict file. These will be used to load and
process images extracted from a bag file.
They would require "ad hoc" dataloader to prevent out of memory errors.
I decided to create two separate files instead of modifying the original ones.
'''

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="8097")
parser.add_argument("--train", action='store_true')
parser.add_argument("--predict", action='store_true')
opt = parser.parse_args()

if opt.predict:
    for i in range(1):
        os.system("python predict_2.py \
                --dataroot ./test_dataset \
                --name enlightening \
                --model single \
                --which_direction AtoB \
                --no_dropout \
                --dataset_mode unaligned \
                --which_model_netG sid_unet_resize \
                --skip 1 \
                --use_norm 1 \
                --use_wgan 0 \
                --self_attention \
                --times_residual \
                --instance_norm 0 --resize_or_crop='no'\
                --which_epoch " + str(200 - i * 5))

