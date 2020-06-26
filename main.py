from __future__ import print_function

import os
import logging
import random

import torch

from model import ResNet
from train import *


if __name__ == "__main__":
    args = {
        'epoches': 60,
        'batch_size': 32,
        'lr': 1e-3,
        'bag_size': 20,
        'do_train': False,
        'do_eval': False,
        'do_predict': True,
        'steps_save_ckpt': 10,
        'dataset_dir': '/chpc/home/stu-jrlu-a/ml_dataset/',
        'init_checkpoint': 'resnet18-init.ckpt,
        'output_dir': '/chpc/home/stu-jrlu-a/ml_dataset/'
    }
    if args['do_train']:
        model = ResNet(BasicBlock, [2,2,2,2], 512, 512)
        train_model(args, model)
    if args['do_eval']:
        model= torch.load(args['output_dir'] + args['init_checkpoint'])
        eval(args, model)
    if args['do_predict']:
        model= torch.load(args['output_dir'] + args['init_checkpoint'])
        predict(args, model)


    


            