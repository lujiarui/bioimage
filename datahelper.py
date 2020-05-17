"""
    Dataset loader.
"""

from os import listdir
import numpy as np
import warnings
from PIL import Image
import csv

warnings.filterwarnings('ignore')



# Read the csv, output: dict{ dirname(str) -> labels(list) }
def load_csvLabel(filename):
    """
    Args:
        filename: /path/to/file/ofcsvlabel
    Return:
        
    """
    dirname_label_map = {}
    with open(filename,'r') as cfile:
        reader = csv.reader(cfile)
        for row in reader:
            name = row[0]
            labels = (row[1]).split(';')
            dirname_label_map[name] = labels
    return dirname_label_map



def load_Dataset(directory_name):
    """
    Args:
        directory_name: The directory where the dataset locate
        For example: /Users/apple/Downloads/ml_dataset, with last / trimmed
                    And /Users/apple/Downloads/ml_dataset/train/xxxxxxxxx/ should contain the images
    Return:
        training_samples: List of ndarray[shape=(512 x 512 x 3)]
        labels: List of labelset(list) 
    """
    if directory_name[-1] == '/':
        raise IOError("Please trim the last / as the directory name")

    if not directory_name:
        raise IOError("Empty input!")

    trainingBatches     = listdir(directory_name + "/train/")
    trainingImages      = {dirname: listdir(directory_name + "/train/{}".format(dirname)) for dirname in trainingBatches}
    dir2label           = load_csvLabel(directory_name + "/train.csv")    # dict: dirname(str) -> labels(list)
    
    print("Load file path done...")

    training_size = 0 # trainset size
    for imageList in trainingImages.values():
        training_size += len(imageList)
    training_samples = []
    labels = [] # same size as m, list of str '1' as elements
    count = 0   # index 
    print("Start loading the Image and convering... \nTotally %d images as dataset"% training_size)

    for dirname, imageList in trainingImages.items():
        for image in imageList:
            labels.append(dir2label[dirname])
            # image file -> ndarray as (512,512,3), each as uint8 since the channel range from [0,255]
            training_samples.append(np.array( Image.open( "./train/%s/%s" % (dirname,image) ), dtype=np.uint8))
            count += 1
            print("Loading Image: %.2f" % (count/training_size), end='\r', flush=True)
    return training_samples, labels


if __name__ == "__main__":
    dm, labels = load_Data()
    print(len(dm), len(labels))

