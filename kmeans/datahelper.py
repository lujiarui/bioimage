"""
    Dataset loader.
"""

from os import listdir,chdir
import warnings
import csv
from random import shuffle
import logging

import torch

warnings.filterwarnings('ignore')


def load_label(filename):
    """
    process the .csv label file
    Args:
        filename: /path/to/file/of/csvlabel/ (csv)
    Return:
        dirname_label_map: a hashmap {dirname <--> label} (dict: str->list[int])
    """
    dirname_label_map = {}
    with open(filename,'r') as cfile:
        reader = csv.reader(cfile)
        for row in reader:
            name = row[0]
            label = (row[1]).split(';')
            int_label = [int(lab) for lab in label]
            dirname_label_map[name] = int_label
    return dirname_label_map



def load_dataset(path_to_trainset, dev_ratio=0.12, num_epoch=40):
    """Generator, generate the partitioned training set and dev set for an epoch
    Args:
        path_to_trainset: The directory where the training dataset locate
            For example: /Users/apple/Downloads/ml_dataset/train/
        dev_ratio: the ratio of dev set in the total set
        num_epoch: the number of epoches for loop
    Return:(generator)
        training_set: list of directory_names for train (str in list)
        dev_set: list of directory_names for eval (str in list)
    """
    if not path_to_trainset:
        raise IOError("[ERROR] Empty input!")
    total_sets  = listdir(path_to_trainset)    # list
    
    #dir2label           = load_csvLabel(directory_name + "/train.csv")    # dict: dirname(str) -> labels(list)
    logging.info(' Load directory path done.')
    training_size = int(len(total_sets) * (1 - dev_ratio)) # train set size
    for epoch in range(num_epoch):
        # shuffle(total_sets) ==> generate a partiton: trainingset and devset
        shuffle(total_sets)
        training_set = total_sets[:training_size]
        dev_set = total_sets[training_size:]
        yield training_set, dev_set
    



