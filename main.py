from __future__ import print_function

import argparse
import os
import logging
import random

import torch
import torch.nn as nn

from model import ConvNN
from train import train_model




# Parser
parser = argparse.ArgumentParser(description='Machine-Learning-parameters')
parser.add_argument('--lr', type=float, default=3e-5,
                    help='learning rate (default: 3e-5')
parser.add_argument('--num-epoches', type=int, default=3,
                    help='number of training epoches')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training/eval/predict batch size')
parser.add_argument('--do-train', default=True,
                    help='do training subroutine')
parser.add_argument('--do-predict',default=False,
                    help='load local checkpoint and do predict')
parser.add_argument('--init-checkpoint', type=str, default=None,
                    help='load model ckpt and predict or train')
parser.add_argument('--path-to-trainset', type=str, default=None,
                    help='path to the training dataset')
parser.add_argument('--path-to-testset', type=str, default=None,
                    help='path to the predict/test dataset')        
parser.add_argument('--output-dir', type=str, default=None,
                    help='path to the directory where output file located')              
parser.add_argument('--save-checkpoint-steps', type=int, default=500,
                    help='save model checkpoint per n steps')



if __name__ == "__main__":
    args = parser.parse_args()
    model = ResNet(BasicBlock, [2,2,2,2], 512, 512)
    
    # Simulate the train-subroutine
    # COPY from datahelper
    if args.do_train:
        train_model(args, model)
    
    if args.do_predict:
        predict(args)
    


            