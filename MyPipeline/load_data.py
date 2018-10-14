'''
# # Загрузка тестовых данных
'''
from local_config import basicpath

path_train = basicpath + 'train/'
path_test = basicpath + 'test/'


import sys
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm_notebook

def load_image(path, mask = False, to_gray=False):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)
    
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path))

    if mask:
        # Convert mask to 0 and 1 format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img // 255).astype(np.float32)
        return img
    else:
        if to_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = (img / 255.0).astype(np.float32)
        if to_gray:
            img = img.reshape((img.shape[0],img.shape[1],1))
        return img
        #torch return  torch.from_numpy(img).float().reshape((img.shape[0],img.shape[1],1)).permute([2, 0, 1])

def LoadImages(df, train_data = True, to_gray=False):
    path = path_train if train_data else path_test
    path_images = path + 'images/'
    path_masks  = path + 'masks/'
    df["images"] = [np.array(load_image(path_images+"{}.png".format(idx), to_gray)) for idx in tqdm_notebook(df.index)]
    if train_data:
        df["masks"] = [np.array(load_image(path_masks+"{}.png".format(idx), mask=True)) for idx in tqdm_notebook(df.index)]

def LoadDataLists(DEV_MODE_RANGE = 0):
    depths_df = pd.read_csv(basicpath+"/depths.csv", index_col="id")
    train_df = pd.read_csv(basicpath+"/train.csv", index_col="id", usecols=[0])
    train_df = train_df.join(depths_df)
    test_df = depths_df[~depths_df.index.isin(train_df.index)]

    folds_df = pd.read_csv(basicpath+"/test_folds.csv", index_col="id")
    train_df = folds_df.join(train_df)

    if DEV_MODE_RANGE:
        train_df = train_df.head(DEV_MODE_RANGE)
        test_df = test_df.head(DEV_MODE_RANGE)
        depths_df = depths_df[depths_df.index.isin(train_df.index) | depths_df.index.isin(test_df.index)]
    print(train_df.shape, test_df.shape, depths_df.shape)
    return (train_df, test_df)

def LoadData(train_data = True, DEV_MODE_RANGE = 0, to_gray=False):
    '''
    -> train_df, test_df
    '''
    train_df, test_df = LoadDataLists(DEV_MODE_RANGE)
    df = train_df if train_data else test_df
    LoadImages(df, train_data, to_gray)
    return df
    
def SplitTrainData(train_df, test_fold_no):
    '''
    -> train_images, train_masks, validate_images, validate_masks
    '''
    train_images = train_df.images[train_df.test_fold != test_fold_no]
    train_masks  = train_df.masks[train_df.test_fold != test_fold_no]
    validate_images = train_df.images[train_df.test_fold == test_fold_no]
    validate_masks  = train_df.masks[train_df.test_fold == test_fold_no]
    return train_images, train_masks, validate_images, validate_masks