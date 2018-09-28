
# coding: utf-8

def RunTest(params):

    # # Params

    # In[ ]:

    DEV_MODE_RANGE = 0 # off


    # In[ ]:

    # In[ ]:


    def params_dict():
        return {x[0]:x[1] for x in vars(params).items() if not x[0].startswith('__')}
    def params_str():
        return '\n'.join([repr(x[0]) + ' : ' + repr(x[1]) + ',' for x in vars(params).items() if not x[0].startswith('__')])
    def params_hash(shrink_to = 6):
        import hashlib
        import json
        return hashlib.sha1(json.dumps(params_dict(), sort_keys=True).encode()).hexdigest()[:shrink_to]
    def params_save(fn, verbose = True):
        params_fn = fn+'.param.txt'
        with open(params_fn, 'w+') as f:
            s = params_str()
            hash = params_hash(shrink_to = 1000)
            s = '{\n' + s + '\n}\nhash: ' + hash[:6] + ' ' + hash[6:]
            f.write(s)
            if verbose:
                print('perams: '+ s + '\nsaved to ' + params_fn)
            


    # # Imports

    # In[ ]:


    import sys
    #sys.path.append(r'D:\Programming\3rd_party\keras')


    # In[ ]:


    import sys
    from imp import reload
    import numpy as np
    import keras
    import datetime
    import time

    from keras.models import Model, load_model
    from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
    from keras.layers.core import Lambda
    from keras.layers.convolutional import Conv2D, Conv2DTranspose
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.merge import concatenate
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
    from keras import backend as K

    import tensorflow as tf


    # # Load data

    # In[ ]:


    import load_data
    load_data = reload(load_data)
    import keras_unet_divrikwicky_model
    keras_unet_divrikwicky_model = reload(keras_unet_divrikwicky_model)


    # In[ ]:


    train_df = load_data.LoadData(train_data = True, DEV_MODE_RANGE = DEV_MODE_RANGE, to_gray = False)


    # In[ ]:


    train_df.images[0].shape


    # In[ ]:


    train_images, train_masks, validate_images, validate_masks = load_data.SplitTrainData(train_df, params.test_fold_no)
    train_images.shape, train_masks.shape, validate_images.shape, validate_masks.shape


    # # Reproducability setup:

    # In[ ]:


    import random as rn

    import os
    os.environ['PYTHONHASHSEED'] = '0'

    np.random.seed(params.seed)
    rn.seed(params.seed)

    #session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(params.seed)
    #sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    sess = tf.Session(graph=tf.get_default_graph())
    K.set_session(sess)


    # # IOU metric

    # In[ ]:


    thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

    def iou(img_true, img_pred):
        assert (img_true.shape[-1]==1) and (len(img_true.shape)==3) or (img_true.shape[-1]!=1) and (len(img_true.shape)==2)
        i = np.sum((img_true*img_pred) >0)
        u = np.sum((img_true + img_pred) >0)
        if u == 0:
            return 1
        return i/u

    def iou_metric(img_true, img_pred):
        img_pred = img_pred > 0.5 # added by sgx 20180728
        if img_true.sum() == img_pred.sum() == 0:
            scores = 1
        else:
            scores = (thresholds <= iou(img_true, img_pred)).mean()
        return scores

    def iou_metric_batch(y_true_in, y_pred_in):
        batch_size = len(y_true_in)
        metric = []
        for batch in range(batch_size):
            value = iou_metric(y_true_in[batch], y_pred_in[batch])
            metric.append(value)
        #print("metric = ",metric)
        return np.mean(metric)

    # adapter for Keras
    def my_iou_metric(label, pred):
        metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float64)
        return metric_value


    # # Data generator

    # In[ ]:


    mean_val = np.mean(train_images.apply(np.mean))
    mean_std = np.mean(train_images.apply(np.std))
    mean_val, mean_std 


    #####################################
    def FillCoordConvNumpy(imgs):
        print(imgs.shape)
        assert len(imgs.shape) == 4
        assert imgs.shape[3] == 3
        n = imgs.shape[2]
        hor_img = np.linspace(mean_val-mean_std, mean_val+mean_std, n).reshape((1, 1,n,1)) 
        n = imgs.shape[1]
        ver_img = np.linspace(mean_val-mean_std, mean_val+mean_std, n).reshape((1, n,1,1)) 
        imgs[:, :, :, 0:1] = hor_img
        imgs[:, :, :, 2:3] = ver_img
    def FillCoordConvList(imgs):
        print(imgs.shape)
        assert len(imgs[0].shape) == 3
        assert imgs[0].shape[2] == 3
        for img in imgs:
            n = img.shape[1]
            hor_img = np.linspace(mean_val-mean_std, mean_val+mean_std, n).reshape((1,n,1)) 
            n = img.shape[0]
            ver_img = np.linspace(mean_val-mean_std, mean_val+mean_std, n).reshape((n,1,1)) 
            img[:, :, 0:1] = hor_img
            img[:, :, 2:3] = ver_img
    
    if params.coord_conv:
        FillCoordConvList(train_images)
        FillCoordConvList(validate_images)
        print (train_images[0][0,0,0], train_images[0][0,0,2])
        assert train_images[0][0,0,0] == mean_val-mean_std
        assert train_images[0][0,0,2] == mean_val-mean_std
    
    ######################################
    
    # In[ ]:


    sys.path.insert(1, '../3rd_party/albumentations')
    sys.path.insert(1, '../3rd_party/imgaug')
    import albumentations


    # In[ ]:


    def basic_aug(p=1.):
        return albumentations.Compose([
            albumentations.Resize(params.augmented_image_size, params.augmented_image_size),
            albumentations.HorizontalFlip(),
            albumentations.RandomCrop(params.nn_image_size, params.nn_image_size),
            albumentations.Normalize(mean = mean_val, std = mean_std*params.norm_sigma_k, max_pixel_value = 1.0),
        ], p=p)


    # In[ ]:


    def more_aug(p=1.):
        return albumentations.Compose([
            albumentations.Resize(params.augmented_image_size, params.augmented_image_size),
            albumentations.RandomScale(0.04),
            albumentations.HorizontalFlip(),
            albumentations.RandomCrop(params.nn_image_size, params.nn_image_size),
            albumentations.Normalize(mean = mean_val, std = mean_std*params.norm_sigma_k, max_pixel_value = 1.0),

            albumentations.Blur(),
            albumentations.Rotate(limit=5),
            albumentations.RandomBrightness(),
            albumentations.RandomContrast(),
        ], p=p)


    # In[ ]:


    import threading
    class AlbuDataGenerator(keras.utils.Sequence):
        'Generates data for Keras'
        def __init__(self, images, masks, batch_size, nn_image_size, shuffle, aug_func):
            'Initialization'
            self.images = images
            self.masks = masks
            self.batch_size = batch_size
            self.nn_image_size = nn_image_size
            self.shuffle = shuffle
            self.indexes = np.arange(len(self.images))
            self.on_epoch_end()
            self.augmentation = aug_func()
            assert len(self.images) >= self.batch_size
            
        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor(len(self.images) / self.batch_size))

        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            X, y = self.__data_generation(indexes)
            return X, y

        def on_epoch_end(self):
            'Updates indexes after each epoch'
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization
            X = np.empty((self.batch_size, self.nn_image_size,self.nn_image_size, params.channels), dtype=np.float32)
            y = np.empty((self.batch_size, self.nn_image_size,self.nn_image_size, 1), dtype=np.float32)

            # Generate data
            for i, index in enumerate(indexes):
                image = self.images[index]
                mask = None if self.masks is None else self.masks[index]
                aug_res = self.augmentation(image = image, mask = mask)
                image = aug_res['image']
                X[i, ...] = image
                mask = aug_res['mask']
                y[i, ...] = mask.reshape(mask.shape[0], mask.shape[1], 1)

            return X, y


    # # model

    # In[ ]:


    sys.path.append('../3rd_party/segmentation_models')
    import segmentation_models
    segmentation_models = reload(segmentation_models)
    from segmentation_models.utils import set_trainable


    # In[ ]:


    model = None
    if params.model == 'FNN':
        model = segmentation_models.FPN(backbone_name=params.backbone, input_shape=(None, None, params.channels),
                                        encoder_weights=params.initial_weightns, freeze_encoder=True)
    if params.model == 'Unet':
        model = segmentation_models.Unet(backbone_name=params.backbone, input_shape=(None, None, params.channels),
                                         encoder_weights=params.initial_weightns, freeze_encoder=True)
    if params.model == 'divrikwicky':
        model = keras_unet_divrikwicky_model.CreateModel(nn_image_size)
        params.backbone = ''


    # In[ ]:


    #model1_file = 'models_1/{model_name}_{backbone_name}_{test_fold_no}.model'.format(model_name=model_name, backbone_name=backbone_name, test_fold_no=test_fold_no)
    model_out_file = 'models_3/{model_name}_{backbone_name}_{optim}_{augw}-{nnw}_lrf{lrf}_ns{norm_sigma_k}_{metric}_{CC}_f{test_fold_no}_{phash}.model'.format(
        model_name=params.model, backbone_name=params.backbone, optim=params.optimizer,
        augw = params.augmented_image_size, nnw = params.nn_image_size, lrf = params.ReduceLROnPlateau['factor'],
		norm_sigma_k=params.norm_sigma_k, metric = params.monitor_metric[0], CC = 'CC' if params.coord_conv else '',
        test_fold_no=params.test_fold_no, phash = params_hash())
    params_save(model_out_file, verbose = True)
    log_out_file = model_out_file+'.log.csv'
    now = datetime.datetime.now()
    print('model:   ' + model_out_file + ' started at ' + now.strftime("%H:%M")) 


    # In[ ]:


    #model = load_model(model1_file, custom_objects={'my_iou_metric': my_iou_metric}) #, 'lavazs_loss': lavazs_loss


    # # Train

    # In[ ]:


    model.compile(loss="binary_crossentropy", optimizer=params.optimizer, metrics=["acc", my_iou_metric]) #, my_iou_metric


    # In[ ]:


    train_gen = AlbuDataGenerator(train_images, train_masks, batch_size=params.batch_size,
                                  nn_image_size = params.nn_image_size, aug_func = basic_aug, shuffle=True)
    val_gen = AlbuDataGenerator(validate_images, validate_masks, batch_size=params.test_batch_size,
                                nn_image_size = params.nn_image_size, aug_func = basic_aug, shuffle=True)


    # In[ ]:


    sys.path.append('../3rd_party/keras-tqdm')
    from keras_tqdm import TQDMCallback, TQDMNotebookCallback


    # In[ ]:

    start_t = time.clock()

    early_stopping = EarlyStopping(monitor=params.monitor_metric[0], mode = params.monitor_metric[1], verbose=1, **params.EarlyStopping)
    model_checkpoint = ModelCheckpoint(model_out_file,
                                       monitor=params.monitor_metric[0], mode = params.monitor_metric[1], save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor=params.monitor_metric[0], mode = params.monitor_metric[1], verbose=1, **params.ReduceLROnPlateau)

    '''
    def get_callbacks(filepath, patience=2):
        es = EarlyStopping('val_loss', patience=patience, mode="min")
        msave = ModelCheckpoint(filepath + '.hdf5', save_best_only=True)
        csv_logger = CSVLogger(filepath+'_log.csv', separator=',', append=False)
        return [es, msave, csv_logger]
    '''
        
    if params.epochs_warmup:
      history = model.fit_generator(train_gen,
                        validation_data=val_gen, 
                        epochs=params.epochs_warmup,
                        callbacks=[early_stopping, model_checkpoint, reduce_lr, TQDMNotebookCallback(leave_inner=True),
                                  CSVLogger(log_out_file, separator=',', append=False)],
                        validation_steps=len(val_gen)*3,
                        workers=5,
                        use_multiprocessing=False,
                        verbose=0)

    set_trainable(model)

    history = model.fit_generator(train_gen,
                        validation_data=val_gen, 
                        epochs=params.epochs,
                        initial_epoch = params.epochs_warmup,
                        callbacks=[early_stopping, model_checkpoint, reduce_lr, TQDMNotebookCallback(leave_inner=True),
                                  CSVLogger(log_out_file, separator=',', append=True)],
                        validation_steps=len(val_gen)*3,
                        workers=5,
                        use_multiprocessing=False,
                        verbose=0
                        )


    # In[ ]:

    print(params_str())
    print('done:   ' + model_out_file)
    print('elapsed: {}s ({}s/iter)'.format(time.clock() - start_t, (time.clock() - start_t)/len(history.epoch) ))

    return model

