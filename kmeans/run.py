import os

from cluster import *
from datahelper import *
from util import *


PATH = '/Users/apple/Downloads/ml_dataset/'

def train():
    """Training subroutine
    """
    label_map = load_label(PATH + 'train.csv')
    train_set = os.path.join(PATH, 'train')
    test_set = os.path.join(PATH, 'test')

    num_words = 10  # num of centers/labels
    dirs = os.listdir(train_set)
    train_img = []
    for dirname in dirs:    # for each directory of images
        train_path = os.path.join(train_set, dirname)
        img_paths = get_img_path(train_path)
        train_img.extend(img_paths)
    
    # get centers coordinate and sift descrition of images
    centers, des_list = getClusterCenters(img_paths=train_img, num_words=num_words, dataset_matrix=None)
    # get all feature vectors
    img_features = get_all_features(des_list=des_list,num_words=num_words, centers=centers)
    
    # test the new sample
    os.listdir(test_set)
    for dirname in dirs:
        print('=== [INFO] Test for dir: ===', dirname)
        images = os.listdir(test_set, dirname)
        for img in images:
            target= os.path.join(test_set,dirname,img)
            retrieval_img(target, img_features, centers, train_img)
 