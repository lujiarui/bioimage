import cv2
import numpy as np
import os
import csv
from random import shuffle
from sklearn.cluster  import KMeans
from matplotlib import pyplot as plt


"""
Image cluster based on SIFT, BOW
    1、use SIFT to extract the feature point for each image
    2、use cluster(KMEANS) to obtain the vision word center, and construct the dictionary of these words
    3、map each [feature point] to the vision [word center] --> feature vector of image
    4、find k-nearest neighborhoods
"""



def getClusterCenters(img_paths, dataset_matrix, num_words):
    """Get Cluster Centers
    Args:
        img_paths: all the images(name) under given directory (complete path)
        dataset_matrix：matrix representation of    
            tips：img_paths, dataset_matrix only need ONE param, equivalently
        num_words: #number of centers
    """
    des_list=[]  # feature description of SIFT(before feature vector)
    des_matrix = np.zeros((1, 128))
    sift_det = cv2.xfeatures2d.SIFT_create()
    if img_paths != None:
        for path in img_paths:
            img = cv2.imread(path)
            # img = cv2.resize(img, (400, 400))
            #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            kp,des = sift_det.detectAndCompute(img,None)
            if des is not None:
                des_matrix = np.vstack((des_matrix,des))
                des_list.append(des)
    elif dataset_matrix!=None:
        for gray in range(dataset_matrix.shape[0]):
            kp,des = sift_det.detectAndCompute(gray,None)
            if des != []:
                des_matrix=np.row_stack((des_matrix,des))
                des_list.append(des)
    else:
        raise ValueError('Illegal input')
    
    des_matrix=des_matrix[1:,:]   # the des matrix of sift
 
    # Compute the kmeans center, construct the dictionary
    kmeans = KMeans(n_clusters=num_words, random_state=27)
    kmeans.fit(des_matrix)
    centers = kmeans.cluster_centers_  
    
    return centers, des_list
 

def des2feature(des, num_words, centers):
    """ Transform the feature description (des) of each image ==> feature vector
        des: feature description of an image
        num_words: number of vision word/ kmeans center
        centers: coordinate of each centers  : num_words*128
    Returns: 
    feature vector : 1*num_words
    """
    img_feature_vec = np.zeros((1,num_words),'float32')
    for i in range(des.shape[0]):
        feature_k_rows = np.ones((num_words,128),'float32')
        feature = des[i]
        feature_k_rows = feature_k_rows*feature
        feature_k_rows = np.sum((feature_k_rows - centers)**2,1)
        index = np.argmax(feature_k_rows)
        img_feature_vec[0][index] += 1
    return img_feature_vec
 
def get_all_features(des_list,num_words, centers):
    """Get all the features for given des_list
    """
    allvec = np.zeros((len(des_list),num_words), 'float32')
    for i in range(len(des_list)):
        if des_list[i]!=[]:
            allvec[i] = des2feature(centers=centers,des=des_list[i],num_words=num_words)
    return allvec
 
def getNearestImg(feature, dataset, num_close):
    """Find nearest neighbors for given feature
        feature: feature of test image
        dataset: image training datasets
        num_close: number of neighbors
    Returns:
        Nearest images
    """
    features = np.ones((dataset.shape[0], len(feature)), 'float32')
    features = features * feature
    # Euclidean distance
    dist = np.sum((features-dataset)**2, 1)
    dist_index = np.argsort(dist)
    return dist_index[:num_close]
 


def retrieval_img(img_path,img_dataset,centers,img_paths):
    """Find nearest neighbors for given image
    Args:
        img: test image
        img_dataset: dataset (matrix)
        centers: cluster centers
        img_paths: all the image paths
    """
    # number of neighbors
    num_close = 9
    img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift_det = cv2.xfeatures2d.SIFT_create()
    kp,des = sift_det.detectAndCompute(img, None)
    feature = des2feature(des=des, centers=centers, num_words=num_words)

    sorted_index = getNearestImg(feature, img_dataset, num_close)
    
    showImg(img_path,sorted_index,img_paths)
 




