import cv2
import os
from matplotlib import pyplot as plt



def get_img_path(training_path):
    """Get all the images(for training) from directory <training_path>
    """
    training_names = os.listdir(training_path)
    # all the images with following format
    pic_names=['jpg']
    for name in training_names:
        file_format=name.split('.')[-1]
        if file_format not in pic_names:
            training_names.remove(name)
 
    img_paths=[]   # all the images(name) under given directory
    for name in training_names:
        img_path = os.path.join(training_path,name)
        img_paths.append(img_path)
    return img_paths

 
def showImg(target_img_path, index, dataset_paths):
    """Show the nearest neighbors
    target_img: test images
    dataset_paths：all the images for training dataset
    """
    # get img path
    paths=[]
    for i in index:
        paths.append(dataset_paths[i])
        
    plt.figure(figsize=(8,10))    #  figsize for display
    plt.subplot(432),plt.imshow(plt.imread(target_img_path)),plt.title('target_image')
    
    for i in range(len(index)):
        plt.subplot(4,3,i+4),plt.imshow(plt.imread(paths[i]))
    plt.show()
 