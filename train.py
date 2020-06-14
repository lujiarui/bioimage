import argparse
import os
import logging
import random

import torch
import torch.nn as nn
from sklearn import metrics
import matplotlib.pyplot as plt

from datahelper import *
from util import save_training_checkpoint
from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Device configuration
print("[INFO] Utilized device as [{}]".format(device))
batch_size = 32
learning_rate = 1e-5

def train_model(args, model):
    """The model training subroutine
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    path_to_trainset = '/Users/ari/Downloads/ml_dataset/train/'
    label_map = load_label('/Users/ari/Downloads/ml_dataset/train.csv')
    dataset_loader = load_dataset(path_to_trainset)
    for epoch in range(1,2):    #args.num_epoches
        # generate a dataset(str)
        ts,ds = next(dataset_loader)
        flag = 1
        counter = 1
        while flag: # exploit training set by 'bags' ==> to generate samples with diverse classes
            images,labels,tot = [],[],[]
            # fetch 5 bags of samples and "shuffle" them 
            # Then feed to the NN
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
            
            # Batch training, based on the index partition
            # partition: batch (len=32(default)) starting-index of images ABOVE
            partition = []
            for i in range(0, len(images), batch_size):
                partition.append(i) 
            step = 0    # current 'bags'
            for pt in range(len(partition) - 1):
                print('[INFO] Now do training .. Epoch{} | Bag{} | miniBatch{}'
                    .format(epoch, counter, step))
                # A batch train
                if pt == len(partition) - 1:
                    image_batch, label_batch = torch.cat(images[partition[pt]: ], dim=0), torch.cat(labels[partition[pt]: ],dim=0)
                else: 
                    image_batch, label_batch = torch.cat(images[partition[pt]:partition[pt+1]], dim=0), torch.cat(labels[partition[pt]:partition[pt+1] ],dim=0)
                
                image_batch.to(device)
                out = model(image_batch)

                # To obtain the Gold label(multi-class)
                v_length = len(label_batch)

                #print('[DEBUG]out-shape:{},label-shape:{}'.format(out.shape,label_batch.shape))
                loss = criterion(out.squeeze(), label_batch.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
            print('[INFO] Epoch[{}/{}], Step[{}] | Loss: {:.5f}'
                  .format(epoch, 2, counter, loss.item())) # args.num_epoches
            #if counter % (batch_size * args.save_checkpoint_steps) == 0:
            #    save_training_checkpoint(model, False, counter//batch_size)
            if (counter + 1) % 10 == 0:
                torch.save(model,'model-{}.ckpt'.format(counter))
            counter += 1
            
        # ==== Evaluate this epoch result using dev set ====
        torch.save(model,'epoch-{}.ckpt'.format(epoch))
        images,scores,labels,ytrues = [],[],[],[]
    
        while(ds):
            dirname = ds.pop()
            tmp_images, tmp_labels = unpack_directory(dirname, path_to_trainset, label_map)
            images.extend(tmp_images)
            labels.extend(tmp_labels)
        # Predicted score
        partition = []
        for i in range(0, len(images), batch_size):
            partition.append(i) 
        # minibatch training
        for pt in range(len(partition) - 1):
            # A batch train
            if pt == len(partition) - 1:
                image_batch, label_batch = torch.cat(images[partition[pt]: ], dim=0), labels[partition[pt]: ]
                y_gold = torch.cat(label_batch, dim=0)
            else: 
                image_batch, label_batch = torch.cat(images[partition[pt]:partition[pt+1]], dim=0), labels[partition[pt]:partition[pt+1] ]
                y_gold = torch.cat(label_batch, dim=0)
            image_batch.to(device)
            out = model(image_batch).detach()
            scores.append(out)
            ytrues.append(y_gold)
        # concat
        y_score = torch.cat(scores,dim=0).numpy()
        y_true = torch.cat(ytrues,dim=0)
        y_true = torch.randint(0, 2, y_true.shape).numpy()
        thresh = 0.5
        y_pred = np.array([[1 if i > thresh else 0 for i in j] for j in y_score])
        
        # To obtain the Gold label(multi-class)
        # AUC Eval
        roc_auc = metrics.roc_auc_score(y_true, y_score, average='macro')
        # macro F1 Eval
        f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
        # micro F1 Eval
        f1_micro = metrics.f1_score(y_true, y_pred, average='micro')

        print('[INFO] Eval result: \n| AUC:{} | \n| F1 Macro:{} | \n | F1 Micro:{} |'.format(
            roc_auc, f1_macro, f1_micro))
        # plot the ROC (sklearn does not support multi-label so to average)
        # along the column
        fpr_list, tpr_list = [], []
        for y_t, y_s in zip(y_true.transpose(), y_score.transpose()):
            fpr, tpr, threshold = metrics.roc_curve(y_t, y_s)
            fpr_list.append(fpr)
            tpr_list.append(tpr)

        fpr = np.concatenate(fpr_list)
        tpr = np.concatenate(tpr_list)

        fpr = np.mean(fpr, axis=0)
        tpr = np.mean(tpr, axis=0)

        lw = 2
        plt.figure(figsize=(10,10))
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Evaluation on dev set for protein photography classification')
        plt.legend(loc="lower right")
        plt.show()
        plt.close()
        

def predict():
    model = torch.load(path_to_model)
    pass


if __name__ == "__main__":
    model  = ResNet(BasicBlock, [2,2,2,2], 512, 512)
    # model = AlexNet(10)
    # model = VGG(make_layers(cfg['A'], batch_norm=True))
    train_model(1, model)