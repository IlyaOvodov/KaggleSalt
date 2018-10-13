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
    :param mode: 'more', 'inference', 'inference+flip', 'basic' ,
    '''
    augs_list = []
    assert mode in {'more', 'inference', 'inference+flip', 'basic', }
    assert max(params.augmented_image_size, params.padded_image_size) >= params.nn_image_size
    augs_list += [albumentations.Resize(params.augmented_image_size, params.augmented_image_size), ]
    if params.padded_image_size:
        augs_list += [albumentations.PadIfNeeded(min_height=params.padded_image_size,
                                                 min_width=params.padded_image_size,
                                                 border_mode=cv2.BORDER_REFLECT_101),]
    if mode != 'inference':
        if mode == 'inference+flip':
            augs_list += [albumentations.HorizontalFlip(p=1.), ]
        else:
            augs_list += [albumentations.HorizontalFlip(), ]
    if mode == 'more':
        augs_list += [albumentations.RandomScale(0.1), ]
    if mode in ['inference', 'inference+flip']:
        augs_list += [albumentations.CenterCrop(params.nn_image_size, params.nn_image_size),]
    else:
        augs_list += [albumentations.RandomCrop(params.nn_image_size, params.nn_image_size),]
    augs_list += [albumentations.ToFloat(),
                  albumentations.Normalize(mean=mean[0], std=mean[1] * params.norm_sigma_k,
                                           max_pixel_value=1.0), ]
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
        assert len(self.images) >= self.batch_size, (len(self.images), self.batch_size)

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
            if self.masks is None:
                aug_res = self.augmentation(image=image)
            else:
                mask = self.masks[index]
                aug_res = self.augmentation(image=image, mask=mask)
                mask = aug_res['mask']
                y[i, ...] = mask.reshape(mask.shape[0], mask.shape[1], 1)
            image = aug_res['image']
            X[i, ...] = image

        return X, y

class AlbuDataGeneratorWithPseudoLabelling(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, images, masks, test_images, batch_size, nn_image_size, shuffle, mode, params, mean, use_ceil=False):
        'Initialization'
        assert use_ceil == False # batch mast have fixed length
        self.images = images
        self.masks = masks
        self.test_images = test_images
        assert batch_size % 2 == 0, batch_size
        self.semi_batch_size = batch_size // 2
        self.nn_image_size = nn_image_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.images))
        self.test_indexes = np.arange(len(self.test_images))
        self.on_epoch_end()
        self.augmentation = common_aug(mode, params, mean)
        self.channels = params.channels
        self.use_ceil = use_ceil
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            np.random.shuffle(self.test_indexes)
        assert len(self.images) >= self.semi_batch_size, (len(self.images), self.semi_batch_size)
        assert len(self.test_images) >= self.semi_batch_size, (len(self.test_images), self.semi_batch_size)

    def __len__(self):
        'Denotes the number of batches per epoch'
        func = np.ceil if self.use_ceil else np.floor
        return int(func(len(self.images) / self.semi_batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        X, y = self.__data_generation(index)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            np.random.shuffle(self.test_indexes)

    def __data_generation(self, batch_index):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.semi_batch_size*2, self.nn_image_size, self.nn_image_size, self.channels), dtype=np.float32)
        y = np.zeros((self.semi_batch_size*2, self.nn_image_size, self.nn_image_size, 1), dtype=np.float32)

        # Generate data
        for i in range (self.semi_batch_size*2):
            if i < self.semi_batch_size:
                image = self.images[batch_index * self.semi_batch_size + i]
            else:
                image = self.test_images[batch_index * self.semi_batch_size + i - self.semi_batch_size]
            if (self.masks is None) or (i >= self.semi_batch_size):
                aug_res = self.augmentation(image=image)
            else:
                mask = self.masks[batch_index * self.semi_batch_size + i]
                aug_res = self.augmentation(image=image, mask=mask)
                mask = aug_res['mask']
                y[i, ...] = mask.reshape(mask.shape[0], mask.shape[1], 1)
            image = aug_res['image']
            X[i, ...] = image

        return X, y