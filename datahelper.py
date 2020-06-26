"""
    Dataset loader.
"""

from os import listdir,chdir
import warnings
import csv
from random import shuffle
import logging

import cv2
from PIL import Image
from torchvision import transforms
import torch

warnings.filterwarnings('ignore')

loader = transforms.Compose([transforms.ToTensor()])
unloader = transforms.ToPILImage()

def load_label(filename):
    """
    process the .csv label file
    Args:
        filename: /path/to/file/of/csvlabel/ (csv)
    Return:
        dirname_label_map: a hashmap {dirname <--> label} (dict: str->list[int])
    Example:
        {..., INDEX10097: [3, 7] ,}
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



def load_dataset(path_to_trainset, dev_ratio=0.08, num_epoch=40):
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
    


def unpack_directory(dirname, path_to_trainset, label_map=None):
    """directory_name -> images with label
    Each photo in the same bag has SAME label
    str -> 
        list of torchTensor(unsqueezed at dim=0), list of torchTensor(unsqueezed at dim=0)
        *Reversable by unloader
    Args:
        str(local),str(shared),dict(shared)
    Examples:
        unpack_directory('ENSG00000001630','/Users/apple/Downloads/ml_dataset/train/', label_map)
    Returns:
        raw_images(Tensors), labels(Tensors)
    """
    chdir(path_to_trainset)

    training_samples  = listdir(dirname)
    raw_images = []
    labels = []
    sample_size = len(training_samples)
    if not label_map:
        for sample in training_samples:
        # imagefiles --> (float but not defined) tensors
            image = Image.open("./%s/%s" % (dirname,sample)).convert('RGB')
            image = loader(image).unsqueeze(0)  # dim: 0
            raw_images.append(image)
        return raw_images
    for sample in training_samples:
        gold_label = label_map[dirname]
        # Float format
        label = torch.zeros((10),dtype=torch.float32)
        for index in gold_label:
            label[index] = 1.
        label = label.unsqueeze(0)  # dim: 0
        labels.append(label)
        # imagefiles --> (float but not defined) tensors
        image = Image.open("./%s/%s" % (dirname,sample)).convert('RGB')
        image = loader(image).unsqueeze(0)  # dim: 0
        raw_images.append(image)
    return raw_images, labels

def csv_record(filename, rows):
    """One time write
    Args:
        rows: python list
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in rows:
            writer.writerow(row)

