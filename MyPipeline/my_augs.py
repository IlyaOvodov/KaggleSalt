'''
My augmentations
'''
import sys
import numpy as np
sys.path.insert(1, '../3rd_party/albumentations')
sys.path.insert(1, '../3rd_party/imgaug')
import albumentations
import keras
import cv2

def common_aug(mode, params, mean, p=1.):
    '''
    :param mode: 'more', 'inference', 'basic' ,
    '''
    augs_list = []
    assert mode in {'more', 'inference', 'basic', }
    assert max(params.augmented_image_size, params.padded_image_size) >= params.nn_image_size
    augs_list += [albumentations.Resize(params.augmented_image_size, params.augmented_image_size), ]
    if params.padded_image_size:
        augs_list += [albumentations.PadIfNeeded(min_height=params.padded_image_size,
                                                 min_width=params.padded_image_size,
                                                 border_mode=cv2.BORDER_REFLECT_101),]
    if mode != 'inference':
        augs_list += [albumentations.HorizontalFlip(), ]
        if mode == 'more':
            augs_list += [albumentations.RandomScale(0.04), ]
    if mode != 'inference':
        augs_list += [albumentations.RandomCrop(params.nn_image_size, params.nn_image_size),]
    else:
        augs_list += [albumentations.CenterCrop(params.nn_image_size, params.nn_image_size), ]
    augs_list += [albumentations.ToFloat(),
                  albumentations.Normalize(mean=mean[0], std=mean[1] * params.norm_sigma_k,
                                           max_pixel_value=1.0), ]
    if mode != 'inference':
        if mode == 'more':
            augs_list += [albumentations.Blur(),
                         # albumentations.Rotate(limit=5),
                         albumentations.RandomBrightness(),
                         albumentations.RandomContrast(),
                         ]
    return albumentations.Compose(augs_list, p=p)


# import threading
class AlbuDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, images, masks, batch_size, nn_image_size, shuffle, mode, params, mean, use_ceil=False):
        'Initialization'
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.nn_image_size = nn_image_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images))
        self.on_epoch_end()
        self.augmentation = common_aug(mode, params, mean)
        self.channels = params.channels
        self.use_ceil = use_ceil
        assert len(self.images) >= self.batch_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        func = np.ceil if self.use_ceil else np.floor
        return int(func(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.nn_image_size, self.nn_image_size, self.channels), dtype=np.float32)
        y = np.empty((self.batch_size, self.nn_image_size, self.nn_image_size, 1), dtype=np.float32)

        # Generate data
        for i, index in enumerate(indexes):
            image = self.images[index]
            mask = None if self.masks is None else self.masks[index]
            aug_res = self.augmentation(image=image, mask=mask)
            image = aug_res['image']
            X[i, ...] = image
            mask = aug_res['mask']
            y[i, ...] = mask.reshape(mask.shape[0], mask.shape[1], 1)

        return X, y


