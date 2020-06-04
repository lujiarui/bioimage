from __future__ import print_function

import argparse
import os
import logging
import random

import torch
import torch.nn as nn

from model import ConvNN
from datahelper import *

# Hyper-parameters for nn
input_channel = 3
input_size = 512
output_size = 10
batch_size = 16
learning_rate = 3e-5

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






if __name__ == "__main__":
    args = parser.parse_args()
    model = ConvNN(input_channel, input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("[INFO] Utilized device as [{}]".format(device))

    # Simulate the train-subroutine
    # COPY from datahelper
    counter = 0
    for epoch in range(args.num_epoches):
        path_to_trainset = '/Users/ari/Downloads/ml_dataset/train/'
        dataset_loader = load_dataset(path_to_trainset)
        # generate a dataset
        ts,ds = next(dataset_loader)
        
        label_map = load_label('/Users/ari/Downloads/ml_dataset/train.csv')
        flag = 1
        while flag: # exploit training set
            images,labels = [],[]
            
            # fetch 5 classes of samples and shuffle them
            for i in range(5):
                if not ts:
                    break
                    flag = 0
                dirname = ts.pop()
                tmp_images, tmp_labels = unpack_directory(dirname, path_to_trainset,label_map)
                images.extend(tmp_images)
                labels.extend(tmp_labels)
            tot = list(zip(images,labels))
            random.shuffle(tot)
            images[:], labels[:] = zip(*tot)
            
            # partition : batch = 32(default)
            partition = []
            for i in range(0, len(images), batch_size):
                partition.append(i) 
            for pt in range(len(partition) - 1):
                # A batch train
                if pt == len(partition) - 1:
                    image_batch, label_batch = torch.cat(images[partition[pt]: ], dim=0), torch.cat(labels[partition[pt]: ],dim=0)
                else: 
                    image_batch, label_batch = torch.cat(images[partition[pt]:partition[pt+1]], dim=0), torch.cat(labels[partition[pt]:partition[pt+1] ],dim=0)
                image_batch.to(device)
                label_batch.to(device)
                out = model(image_batch)
                #print('[DEBUG]out-shape:{},label-shape:{}'.format(out.shape,label_batch.shape))
                loss = criterion(out.squeeze(), label_batch.squeeze())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if counter % 8*5 == 0:
                    print('[INFO] Epoch[{}/{}], Step[{}] | Loss: {:.4f}'
                                .format(epoch+1, args.num_epoches,counter, loss.item()))
                counter += batch_size


            