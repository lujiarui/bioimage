import argparse
import os
import logging
import random
import time

import torch
import torchvision
import torch.nn as nn
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix

from datahelper import *
from model import *
from util import plot

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Device configuration
print("[INFO] Utilized device as [{}]".format(device))



def train_model(args, model):
    """The model training subroutine, including epoch-wise eval
    """
    # deploy the model to device if avail
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50], gamma=0.1)
    
    path_to_trainset = args['dataset_dir'] + 'train/'
    label_map = load_label(args['dataset_dir'] + 'train.csv')
    dataset_loader = load_dataset(path_to_trainset)
    # generate a dataset(str_dirname)
    train_set, dev_set = next(dataset_loader)
    
    tot_loss = []
    tot_devacc = []

    for epoch in range(1, args['epoches'] + 1):    #args.num_epoches
        ts = train_set[:]
        random.shuffle(ts)
        ds = dev_set[:]
        random.shuffle(ds)
        start_time = time.time()
        flag = 1
        counter = 1
        epoch_loss = 0
        while flag: # exploit training set by 'bags' ==> to generate samples with diverse classes
            images,labels,tot = [],[],[]
            # fetch 20 bags of samples and "shuffle" them 
            # Then feed to the NN
            for i in range(args['bag_size']):
                if not ts:
                    flag = 0
                    break
                dirname = ts.pop()
                tmp_images, tmp_labels = unpack_directory(dirname, path_to_trainset,label_map)
                images.extend(tmp_images)
                labels.extend(tmp_labels)
            tot = list(zip(images,labels))
            random.shuffle(tot)
            if tot == []:
                break
            images[:], labels[:] = zip(*tot)
            
            # Batch training, based on the index partition
            # partition: batch (len=32(default)) starting-index of images ABOVE
            partition = []
            for i in range(0, len(images), args['batch_size']):
                partition.append(i) 
            step = 0    # current 'bags'
            for pt in range(len(partition)):
                #print('[INFO] Now do training .. Epoch{} | Bag{} | miniBatch{}'
                #    .format(epoch, counter, step))
                # A batch train
                if pt == len(partition) - 1:
                    image_batch, label_batch = torch.cat(images[partition[pt]: ], dim=0), torch.cat(labels[partition[pt]: ],dim=0)
                else: 
                    image_batch, label_batch = torch.cat(images[partition[pt]:partition[pt+1]], dim=0), torch.cat(labels[partition[pt]:partition[pt+1] ],dim=0)
                
                image_batch = image_batch.to(device)
                label_batch = label_batch.to(device)
                out = model(image_batch)

                # To obtain the Gold label(multi-class)
                v_length = len(label_batch)

                #print('[DEBUG]out-shape:{},label-shape:{}'.format(out.shape,label_batch.shape))
                loss = criterion(out.squeeze(), label_batch.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                epoch_loss += loss.item()
            
            # Eval step-wise use batch-size in train set
            samples_ = random.sample(range(len(images)), args['batch_size'])    # random sample explored
            sample_img = [images[idx] for idx in samples_]
            sample_img = torch.cat(sample_img, dim=0).to(device)
            sample_label = [labels[idx] for idx in samples_]
            sample_label = torch.cat(sample_label, dim=0)
            s_out = model(sample_img).detach()
            s_out = s_out.cpu()

            thresholds = (s_out.max(dim=0).values + s_out.min(dim=0).values) / 2
            hard_label = np.array([[1 if score > thresholds[i] else 0 for i, score in enumerate(j)] for j in s_out])

            _tmp = abs(sample_label - hard_label)
            acc = 0
            for row in _tmp:
                _f = 1
                for element in row:
                    if element > 0.0001:
                      _f = 0
                if _f:
                    acc += 1  
            acc = float(acc) / args['batch_size']
            
            current_time = time.time()
            print('[LOGGER] Epoch[{}/{}], Step[{}]| Acc: {:.3f} | Time elapsed: {:.2f}/sec'
                  .format(epoch, args['epoches'], counter, acc, current_time - start_time)) # args.num_epoches

            counter += 1
        tot_loss.append(epoch_loss)
        print('[INFO] Epoch[{}/{}] Ended| Loss {:.4f} | Time elapsed: {:.2f}/sec\nStarting Eval step...'
                  .format(epoch, args['epoches'], epoch_loss, current_time - start_time))
        
        # save model
        if epoch % args['steps_save_ckpt'] == 0:
            torch.save(model, args['output_dir'] + 'epoch-{}.ckpt'.format(epoch))

        # ==== Evaluate this epoch result using dev set ====
        ts = train_set[:]
        devacc = eval(args, model, ts, ds)
        tot_devacc.append(devacc)
        scheduler.step()

    plt.plot(tot_loss)
    plt.ylabel('Moving Loss each training epoches')
    plt.xlabel('Epoches')
    plt.savefig(args['output_dir'] + 'loss.png')
    plt.close()

    plt.plot(tot_devacc)
    plt.ylabel('Moving Acc each training epoches')
    plt.xlabel('Epoches')
    plt.savefig(args['output_dir'] + 'acc.png')
    plt.close()


def eval(args, model, trainset, devset):
    images,scores,labels,xtrues = [],[],[],[]
    path_to_trainset = args['dataset_dir'] + 'train/'
    label_map = load_label(args['dataset_dir'] + 'train.csv')
    ds = devset[:]
    ts = trainset[:]
    
    # train the svm
    while(ts):
        # traverse each dir in dev set
        dirname = ts.pop()
        images, labels = unpack_directory(dirname, path_to_trainset, label_map)
        random.shuffle(images)
        x_gold = labels[0]
        dir_score = []
        # Predicted score
        partition = []
        for i in range(0, len(images), args['batch_size']):
            partition.append(i) 
        # minibatch training
        for pt in range(len(partition)):
            # A batch train
            if pt == len(partition) - 1:
                image_batch = torch.cat(images[partition[pt]: ], dim=0)
            else: 
                image_batch= torch.cat(images[partition[pt]:partition[pt+1]], dim=0)
            image_batch = image_batch.to(device)
            out = model(image_batch).detach()
            out = out.cpu()
            dir_score.append(out)   # consider a bag at a time            
        dir_score = torch.cat(dir_score, dim=0)
        dir_score = dir_score.mean(dim=0)
        scores.append(dir_score)
        xtrues.append(x_gold)
    x_score = torch.stack(scores,dim=0)
    x_true = torch.cat(xtrues,dim=0)
    svm = svm_decision(x_score, x_true)
    
    images,scores,labels,ytrues = [],[],[],[]
    while(ds):
        # traverse each dir in dev set
        dirname = ds.pop()
        images, labels = unpack_directory(dirname, path_to_trainset, label_map)
        random.shuffle(images)
        y_gold = labels[0]
        dir_score = []
        # Predicted score
        partition = []
        for i in range(0, len(images), args['batch_size']):
            partition.append(i) 
        # minibatch training
        for pt in range(len(partition)):
            # A batch train
            if pt == len(partition) - 1:
                image_batch = torch.cat(images[partition[pt]: ], dim=0)
            else: 
                image_batch= torch.cat(images[partition[pt]:partition[pt+1]], dim=0)
            image_batch = image_batch.to(device)
            out = model(image_batch).detach()
            out = out.cpu()
            dir_score.append(out)   # consider a bag at a time            
        dir_score = torch.cat(dir_score, dim=0)
        dir_score = dir_score.mean(dim=0)
        scores.append(dir_score)
        ytrues.append(y_gold)
    # concat
    y_score = torch.stack(scores,dim=0)
    y_true = torch.cat(ytrues,dim=0)
    # use MID value to represent thresh for each label
    thresholds = (y_score.max(dim=0).values + y_score.min(dim=0).values) / 2
    # To obtain the Gold label(multi-class)
    #y_pred = torch.FloatTensor([[1 if score > thresholds[i] else 0 for i, score in enumerate(j)] for j in y_score])
    y_pred = svm.predict(y_score)
    # Acc record
    diff = y_pred - y_true.numpy()
    devacc = 0
    for row in diff:
        _f = 1
        for element in row:
            if abs(element.item()) > 0.0001:
              _f = 0
        if _f:
            devacc += 1  
    devacc = float(devacc) / len(y_true)
    # plot roc curve
    plot(y_score, y_true, args['output_dir'])
    # macro F1 Eval
    f1_macro = metrics.f1_score(y_true, y_pred, average='macro')
    # micro F1 Eval
    f1_micro = metrics.f1_score(y_true, y_pred, average='micro')
    # AUC Eval
    try:
        roc_auc = metrics.roc_auc_score(y_true, y_score, average='macro')
    except ValueError:
        print('[WARNING] Current dev set has not all of the labels')
        roc_auc = -1 
    print('[INFO] Eval result:\n|ACC:{}|\n|AUC:{}|\n|F1 Macro:{}|\n|F1 Micro:{}|'.format(
        devacc, roc_auc, f1_macro, f1_micro))
    return devacc   # for train subroutine
    

def predict(args, model, trainset):
    model = model.to(device)
    
    path_to_testset = args['dataset_dir'] + 'test/'
    test_sets  = listdir(path_to_testset)
    
    path_to_trainset = args['dataset_dir'] + 'train/'
    label_map = load_label(args['dataset_dir'] + 'train.csv')
    
    # ===== train the svm =====
    images,scores,labels,xtrues = [],[],[],[]
    while trainset:
        # traverse each dir in dev set
        dirname = trainset.pop()
        images, labels = unpack_directory(dirname, path_to_trainset, label_map)
        random.shuffle(images)
        x_gold = labels[0]  # directory share the label
        # Predicted score
        dir_score = []
        partition = []  # minibatch training
        for i in range(0, len(images), args['batch_size']):
            partition.append(i) 
        # minibatch training
        for pt in range(len(partition)):
            # A batch train
            if pt == len(partition) - 1:
                image_batch = torch.cat(images[partition[pt]: ], dim=0)
            else: 
                image_batch= torch.cat(images[partition[pt]:partition[pt+1]], dim=0)
            image_batch = image_batch.to(device)
            out = model(image_batch).detach()
            out = out.cpu()
            dir_score.append(out)   # consider a bag at a time            
        dir_score = torch.cat(dir_score, dim=0)
        dir_score = dir_score.mean(dim=0)
        scores.append(dir_score)
        xtrues.append(x_gold)
    x_score = torch.stack(scores, dim=0).numpy()
    x_true = torch.cat(xtrues, dim=0).numpy()
    print('Training set for SVM: ', x_score.shape)
    svm = svm_decision(x_score, x_true)
    
    # ===== predict the score =====
    y_score = []
    for dirname in test_sets:
        # predict for each file
        images = unpack_directory(dirname, path_to_testset)
        dir_score = []
        partition = []
        for i in range(0, len(images), args['batch_size']):
            partition.append(i) 
        for pt in range(len(partition)):
            # minibatch train
            if pt == len(partition) - 1:
                image_batch = torch.cat(images[partition[pt]: ], dim=0)
            else: 
                image_batch = torch.cat(images[partition[pt]:partition[pt+1]], dim=0)
            image_batch = image_batch.to(device)
            out = model(image_batch).detach()
            out = out.cpu()
            dir_score.append(out)
            
        dir_scores = torch.cat(dir_score, dim=0)
        if len(images) != dir_scores.shape[0]:
            print('[WARNING] The read and write are not matched.')
        dir_scores = dir_scores.mean(dim=0) # reduce dim=0 (shape=10)
        y_score.append(dir_scores)
    # row represents each dir
    # column represents each label
    y_score = torch.stack(y_score, dim=0)
    y_prob = y_scores.numpy().round(4)  # output, round=4
    
    #thresholds = (y_scores.max(dim=0).values + y_scores.min(dim=0).values) / 2
    #str_label = [[str(i) for i, score in enumerate(_scores) if score > thresholds[i]] for _scores in y_scores]

    y_pred = svm.predict(y_score)
    str_label = [[str(i) for i, pred_label in enumerate(row) if pred_label >= 0.99] for row in y_pred]  # >=0.99 ~ ==1
    str_prob = [[str(p) for p in list(_prob)] for _prob in y_prob]

    # split using ;
    print_score = [[dirname, ';'.join(_prob)] for dirname, _prob in zip(test_sets, str_prob)]
    print_label = [[dirname, ';'.join(_label)] for dirname, _label in zip(test_sets, str_label)]

    csv_record(args['output_dir'] + 'test_pred.csv', print_score)
    csv_record(args['output_dir'] + 'test.csv', print_label)
    print('[INFO] Predict done.')


def svm_decision(y_score, y_true):
    """
    Args:
        y_score: [batch x 10] score of each label for batch samples
        y_true: [batch x 10] labels
    Return: 
        clf(svm classifier)
    """
    # Due to the imbalance of dataset
    clf = OneVsRestClassifier(SVC(class_weight='balanced'))
    clf.fit(y_score, y_true)    
    y_pred = clf.predict(y_score)

    return clf


if __name__ == "__main__":
    #model  = ResNet(BasicBlock, [2,2,2,2], 512, 512)
    # model = ResNet(Bottleneck, [3, 4, 6, 3], 512, 512)
    #model = AlexNet(10)
    #predict(model)
    #eval(model)
    args = {
        'epoches': 50,
        'batch_size': 32,
        'lr': 1e-3,
        'bag_size': 20,
        'do_train': True,
        'do_eval': True,
        'do_predict': False,
        'steps_save_ckpt': 5,
        'dataset_dir': '/chpc/home/stu-jrlu-a/ml_dataset/',
        'init_checkpoint': 'resnet18-init.ckpt',
        'output_dir': '/chpc/home/stu-jrlu-a/ml_dataset/'
    }
    #train_model(args, model)

    path_to_trainset = args['dataset_dir'] + 'train/'
    label_map = load_label(args['dataset_dir'] + 'train.csv')
    dataset_loader = load_dataset(path_to_trainset)
    # generate a dataset(str_dirname)
    train_set, dev_set = next(dataset_loader)
    train_set.extend(dev_set)   # feed all the data to train svm
    model = torch.load(args['dataset_dir'] + args['init_checkpoint'])
    #eval(args, model, train_set, dev_set)
    predict(args, model, train_set)
