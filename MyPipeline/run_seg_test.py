
# coding: utf-8
def LoadModelParams(model_name):
    import ast
    with open(model_name + '.param.txt') as f:
        s = f.readlines()
        assert s[-2] == '}\n'
        s = ''.join(s[:-1])
        print(s)
        params = ast.literal_eval(s)
    params = type("params", (object,), params)
    return params

def  RunTestOnSizes(params,
            model_name_template = 'models_3/{model}_{backbone}_{optimizer}_{augmented_image_size}-{padded_image_size}-{nn_image_size}_lrf{lrf}_{metric}_{CC}_f{test_fold_no}_{phash}',
            sizes = [(101,192,128), ] # (202,336,224),
            ):
    for sz in sizes:
        params.augmented_image_size = sz[0]
        params.padded_image_size = sz[1]
        params.nn_image_size = sz[2]
        RunTest(params, model_name_template=model_name_template)

def RunTest(params,
            model_name_template = 'models_3/{model}_{backbone}_{optimizer}_{augmented_image_size}-{padded_image_size}-{nn_image_size}_lrf{lrf}_{metric}_{CC}_f{test_fold_no}_{phash}'
            ):

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
                print('params: '+ s + '\nsaved to ' + params_fn)
            


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
    sys.path.append('../3rd_party/LovaszSoftmax/tensorflow')
    import lovasz_losses_tf as L_loss

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
    params.seed += params.test_fold_no
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
        hor_img = np.linspace(-1., 1., n).reshape((1, 1,n,1))
        n = imgs.shape[1]
        ver_img = np.linspace(-1., 1., n).reshape((1, n,1,1))
        imgs[:, :, :, 0:1] = hor_img
        imgs[:, :, :, 2:3] = ver_img
    def FillCoordConvList(imgs):
        print(imgs.shape)
        assert len(imgs[0].shape) == 3
        assert imgs[0].shape[2] == 3
        for img in imgs:
            n = img.shape[1]
            hor_img = np.linspace(-1., 1., n).reshape((1,n,1))
            n = img.shape[0]
            ver_img = np.linspace(-1., 1., n).reshape((n,1,1))
            img[:, :, 0:1] = hor_img
            img[:, :, 2:3] = ver_img
    
    if params.coord_conv:
        FillCoordConvList(train_images)
        FillCoordConvList(validate_images)
        print (train_images[0][0,0,0], train_images[0][0,0,2])
        assert train_images[0][0,0,0] == -1.
        assert train_images[0][0,0,2] == -1.
    
    ######################################
    
    from my_augs import AlbuDataGenerator

    def lavazs_loss(labels, scores):  # Keras and TF has reversed order of args
        return L_loss.lovasz_hinge(2 * scores - 1, labels, ignore=255, per_image=True)

    def bce_lavazs_loss(labels, scores):  # Keras and TF has reversed order of args
        alpha = 0.1
        return alpha*keras.losses.binary_crossentropy(labels, scores) + (1-alpha)*lavazs_loss(labels, scores)


    # # model

    # In[ ]:


    sys.path.append('../3rd_party/segmentation_models')
    import segmentation_models
    segmentation_models = reload(segmentation_models)
    from segmentation_models.utils import set_trainable


    # In[ ]:
    if not hasattr(params, 'model_params'):
        params.model_params = {}

    if params.load_model_from:
        model = load_model(params.load_model_from,
                           custom_objects={'my_iou_metric': my_iou_metric}
                           )
        print('MODEL LOADED from: ' + params.load_model_from)
    else:
        model = None
        if params.model == 'FNN':
            model = segmentation_models.FPN(backbone_name=params.backbone, input_shape=(None, None, params.channels),
                                            encoder_weights=params.initial_weightns, freeze_encoder=True,
                                            dropout = params.dropout,
                                            **params.model_params)
        if params.model == 'FNNdrop':
            model = segmentation_models.FPNdrop(backbone_name=params.backbone, input_shape=(None, None, params.channels),
                                            encoder_weights=params.initial_weightns, freeze_encoder=True,
                                            dropout = params.dropout,
                                            **params.model_params)
        if params.model == 'Unet':
            model = segmentation_models.Unet(backbone_name=params.backbone, input_shape=(None, None, params.channels),
                                             encoder_weights=params.initial_weightns, freeze_encoder=True,
                                            **params.model_params)
        if params.model == 'Linknet':
            model = segmentation_models.Linknet(backbone_name=params.backbone, input_shape=(None, None, params.channels),
                                                encoder_weights=params.initial_weightns, freeze_encoder=True,
                                            **params.model_params)
        if params.model == 'divrikwicky':
            model = keras_unet_divrikwicky_model.CreateModel(params.nn_image_size,
                                            **params.model_params)
            params.backbone = ''
        assert model

    for l in model.layers:
        if isinstance(l, segmentation_models.fpn.layers.UpSampling2D) or isinstance(l, keras.layers.UpSampling2D):
            print(l)
            if hasattr(l, 'interpolation'):
                print(l.interpolation)
                if hasattr(params, 'model_params') and 'interpolation' in params.model_params:
                    l.interpolation = params.model_params['interpolation']
            else:
                print('qq')

    if hasattr(params, 'kernel_constraint_norm') and params.kernel_constraint_norm:
        for l in model.layers:
            if hasattr(l, 'kernel_constraint'):
                print('kernel_constraint for ', l, ' is set to ',  params.kernel_constraint_norm)
                l.kernel_constraint = keras.constraints.get(keras.constraints.max_norm(params.kernel_constraint_norm))

    # In[ ]:
    #print(model.summary())

    model_out_file = model_name_template.format(
        lrf = params.ReduceLROnPlateau['factor'],
		metric = params.monitor_metric[0],
        CC = 'CC' if params.coord_conv else '',
        **vars(params)) + '_f{test_fold_no}_{phash}'.format(test_fold_no = params.test_fold_no, phash = params_hash())
    now = datetime.datetime.now()
    print('model:   ' + model_out_file + '    started at ' + now.strftime("%Y.%m.%d %H:%M:%S"))

    assert not os.path.exists(model_out_file + '.model')

    params_save(model_out_file, verbose = True)
    log_out_file = model_out_file+'.log.csv'


    # In[ ]:


    #model = load_model(model1_file, ) #, 'lavazs_loss': lavazs_loss


    # # Train

    # In[ ]:

    optimizer=params.optimizer
    if optimizer == 'adam':
        optimizer = keras.optimizers.adam(**params.optimizer_params)
    elif optimizer == 'sgd':
        optimizer = keras.optimizers.sgd(**params.optimizer_params)

    if not hasattr(params, 'loss'):
        params.loss = 'binary_crossentropy'
    loss = params.loss
    if loss == 'bce_lavazs_loss':
        loss = bce_lavazs_loss

    model.compile(loss=loss, optimizer=optimizer, metrics=["acc", my_iou_metric]) #, my_iou_metric


    # In[ ]:

    if params.coord_conv:
        mean = (np.asarray((0.,mean_val,0.)), np.asarray((1.,mean_std,1.)))
    else:
        mean = (mean_val, mean_std)

    train_gen = AlbuDataGenerator(train_images, train_masks, batch_size=params.batch_size, nn_image_size = params.nn_image_size,
                                  mode = params.train_augmentation_mode, shuffle=True, params = params, mean=mean)
    val_gen = AlbuDataGenerator(validate_images, validate_masks, batch_size=params.test_batch_size,nn_image_size = params.nn_image_size,
                                mode = params.test_augmentation_mode, shuffle=False, params = params, mean=mean)


    # In[ ]:


    sys.path.append('../3rd_party/keras-tqdm')
    from keras_tqdm import TQDMCallback, TQDMNotebookCallback


    # In[ ]:

    start_t = time.clock()

    early_stopping = EarlyStopping(monitor=params.monitor_metric[0], mode=params.monitor_metric[1], verbose=1,
                                   **params.EarlyStopping)
    model_checkpoint = ModelCheckpoint(model_out_file + '.model',
                                       monitor=params.monitor_metric[0], mode=params.monitor_metric[1],
                                       save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor=params.monitor_metric[0], mode=params.monitor_metric[1], verbose=1,
                                  **params.ReduceLROnPlateau)
    callbacks = [early_stopping, model_checkpoint, reduce_lr, ]

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
                        callbacks=[TQDMNotebookCallback(leave_inner=True),
                                  CSVLogger(log_out_file, separator=',', append=False)],
                        validation_steps=len(val_gen),
                        workers=5,
                        use_multiprocessing=False,
                        verbose=0)

    set_trainable(model)

    use_cosine_lr = hasattr(params, 'cosine_annealing_params')
    if use_cosine_lr:
        from my_callbacks import CosineAnnealing
        callbacks = [CosineAnnealing(len(train_gen), model_out_file, **params.cosine_annealing_params)]

    history = model.fit_generator(train_gen,
                        validation_data=val_gen, 
                        epochs=params.epochs,
                        initial_epoch = 0, #params.epochs_warmup,
                        callbacks= callbacks + [TQDMNotebookCallback(leave_inner=True),
                                  CSVLogger(log_out_file, separator=',', append=False)],
                        validation_steps=len(val_gen),
                        workers=5,
                        use_multiprocessing=False,
                        verbose=0
                        )


    # In[ ]:

    print(params_str())
    print('done:   ' + model_out_file)
    print('elapsed: {}s ({}s/iter)'.format(time.clock() - start_t, (time.clock() - start_t)/len(history.epoch) ))

    return model

if __name__== "__main__":
    params = {
        'seed': 241075,
        'model': 'FNNdrop',
        'backbone': 'resnet34',
        'initial_weightns': 'imagenet',
        'dropout': 0.3,
        'interpolation': 'nearest',  # 'bilinear',
        'optimizer': 'sgd',
        'optimizer_params': {'momentum': 0.9, 'nesterov': True},
        'cosine_annealing_params': {'min_lr': 1e-05, 'max_lr': 0.02, 'period': 20, 'verbose': 1},
        'augmented_image_size': 101,
        'padded_image_size': 192,
        'nn_image_size': 128,
        'channels': 3,
        'coord_conv': False,
        'norm_sigma_k': 1.0,
        'load_model_from': None,
        'train_augmentation_mode': 'basic',
        'test_augmentation_mode': 'inference',
        'epochs_warmup': 2,
        'epochs': 300,
        'batch_size': 20,
        'test_batch_size': 50,
        'monitor_metric': ('val_my_iou_metric', 'max'),
        'ReduceLROnPlateau': {'factor': 0.5, 'patience': 10, 'min_lr': 1e-05},
        'EarlyStopping': {'patience': 50},
        'test_fold_no': 1,
        'attempt': 0,
        'comment': '',
    }
    params = type("params", (object,), params)
    RunTest(params)