# coding: utf-8
import sys
#sys.path.append(r'D:\Programming\3rd_party\keras')
from imp import reload
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style("white")
import pandas as pd

import keras

from keras.models import Model, load_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import tensorflow as tf
import os

from skimage.transform import resize
import cv2
from tqdm import tqdm_notebook

from run_seg_test import LoadModelParams, LoadModel, CreateModel, CompileModel
from my_augs import AlbuDataGenerator
sys.path.insert(1, '../3rd_party/albumentations')
sys.path.insert(1, '../3rd_party/imgaug')
import albumentations
sys.path.append('../3rd_party/keras-tqdm')
from keras_tqdm import TQDMCallback, TQDMNotebookCallback

mean_val = 0.481577
mean_std = 0.11108

def ResultsFileName(save_results_dir, params_file, model_no, flip, is_test_run):
    fn = save_results_dir + params_file + '.' + str(model_no) + '.results'
    if not is_test_run:
        fn += '.val'
    if flip:
        fn += '.flip'
    return  fn

def PredictResults(test_images, data_dir, params_file, model_no, flip, is_test_run, save_results_dir = None, eval_crop_size = 224):
    assert isinstance(model_no, int)
    params = LoadModelParams(data_dir+params_file)
    params.load_model_from = data_dir+params_file + '.' + str(model_no) + '.model'

    # # model
    model1 = LoadModel(params.load_model_from)
    model = None
    if 'interpolation' in params.model_params and params.model_params['interpolation']=='bilinear':
        print('Rebuilding model to fix BILINEAR problem')
        model = CreateModel(params)
        model.set_weights(model1.get_weights())
    else:
        model = model1
    assert model
    CompileModel(model, params, use_pseudo_labeling=False)
    # # Train evaluation
    params.nn_image_size = eval_crop_size #params.padded_image_size

    augmentation_mode = 'inference+flip' if flip else 'inference'
    test_gen = AlbuDataGenerator(test_images, None, batch_size=params.test_batch_size, nn_image_size = params.nn_image_size,
                                mode = augmentation_mode, shuffle=False, params = params, mean=(mean_val, mean_std),
                               use_ceil = True)
    r = model.predict_generator(test_gen, max_queue_size=10, workers=1, use_multiprocessing=False)
    r = r[:test_images.shape[0], ...] # if ceil, r dim can be higher in because of last batch
    if flip:
        for i in range(r.shape[0]):
            fl = cv2.flip(r[i, ...], 1)
            r[i, ...] = fl[..., np.newaxis]
    start_coord = (params.nn_image_size - params.augmented_image_size)//2
    r_orig = r[:, start_coord : start_coord + params.augmented_image_size, start_coord : start_coord + params.augmented_image_size]
    test_results = []
    for i in range(r.shape[0]):
        test_results += [cv2.resize(r_orig[i, :,:,0], (101,101))]
    for i,im in enumerate(test_images):
        if np.sum(im) == 0:
            test_results[i][...] = 0
    if save_results_dir is not None:
        np.save(ResultsFileName(save_results_dir, params_file, model_no, flip, is_test_run), test_results)
    return test_results