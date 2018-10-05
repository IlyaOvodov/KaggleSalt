import six
import csv
import io
import math
from collections import OrderedDict
from collections import Iterable
import numpy as np
from keras.callbacks import Callback
from keras import backend as K
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import to_list

class EvalLrTest(Callback):
    def __init__(self, filename, val_data, lr_min = 1e-6, lr_max=1, steps=1500, val_period = 1, separator=','):
        self.filename = filename
        self.sep = separator
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr = lr_min
        self.val_period = val_period
        self.epoch = 0
        assert steps > 1
        self.lr_increment = pow((lr_max/lr_min), 1./(steps - 1))
        self._open_args = {}
        self.batch_no = 0
        self.keys = None
        self.writer = None
        self.append_header = True
        workers = 5                     # TODO make a parameter
        max_queue_size = 10             # TODO make a parameter
        self.val_enqueuer = OrderedEnqueuer(val_data, use_multiprocessing=False)
        self.val_enqueuer.start(workers=workers,
                           max_queue_size=max_queue_size)
        self.val_enqueuer_gen = self.val_enqueuer.get()
        self.validation_steps = len(val_data)
        super(EvalLrTest, self).__init__()

    def on_train_begin(self, logs=None):
        self.csv_file = io.open(self.filename, 'w', **self._open_args)
        self.lr = self.lr_min

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        K.set_value(self.model.optimizer.lr, self.lr)
        #if self.verbose > 0:
        if batch == 0: # epoch start
            print('\nEpoch %05d Batch %05d: EvalLrTest setting learning rate to %s.' % (self.epoch, self.batch_no, self.lr))


    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = self.lr
        logs['epoch'] = self.epoch

        if (self.batch_no % self.val_period == 0) or (self.lr > self.lr_max):
            val_outs = self.model.evaluate_generator(
                self.val_enqueuer_gen,
                self.validation_steps,
                workers=0)
            val_outs = to_list(val_outs)
            # Same labels assumed.
            for l, o in zip(self.model.metrics_names, val_outs):
                logs['val_' + l] = o

            def handle_value(k):
                is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
                if isinstance(k, six.string_types):
                    return k
                elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                    return '"[%s]"' % (', '.join(map(str, k)))
                else:
                    return k

            if self.keys is None:
                self.keys = sorted(logs.keys())

            if self.model.stop_training:
                # We set NA so that csv parsers do not fail for this last epoch.
                logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

            if not self.writer:
                class CustomDialect(csv.excel):
                    delimiter = self.sep
                fieldnames = ['batch_no'] + self.keys
                if six.PY2:
                    fieldnames = [unicode(x) for x in fieldnames]
                self.writer = csv.DictWriter(self.csv_file,
                                             fieldnames=fieldnames,
                                             dialect=CustomDialect)
                if self.append_header:
                    self.writer.writeheader()

            row_dict = OrderedDict({'batch_no': self.batch_no})
            row_dict.update((key, handle_value(logs[key])) for key in self.keys)
            self.writer.writerow(row_dict)
            self.csv_file.flush()

        self.lr *= self.lr_increment
        self.batch_no += 1
        if self.lr > self.lr_max:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.val_enqueuer.stop()
        self.csv_file.close()
        self.writer = None

class CosineAnnealing(Callback):
    def __init__(self, batches_per_epoch, filepath = '', save_weights_only = False, min_lr = 1.e-5, max_lr = 2.e-2, period=20, verbose = 0):
        '''
        :param min_lr:
        :param max_lr:
        :param period: epochs
        :param batches_per_epoch:
        '''
        super(CosineAnnealing, self).__init__()
        self.filepath = filepath
        self.save_weights_only = save_weights_only
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.batches_period = period*batches_per_epoch
        self.batches_passed_since_restart = 0
        self.lr = self.max_lr
        self.verbose = verbose
        self.period_no = 0

    def on_train_begin(self, logs=None):
        self.batches_passed_since_restart = 0
        self.lr = self.max_lr
        self.period_no = 0

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        self.lr = self.min_lr + (self.max_lr-self.min_lr)*0.5*(1 +
                  math.cos(self.batches_passed_since_restart / self.batches_period * math.pi))
        K.set_value(self.model.optimizer.lr, self.lr)

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealing setting learning rate to %s.' % (epoch, self.lr))

    def on_batch_end(self, batch, logs=None):
        self.batches_passed_since_restart += 1

    def on_epoch_end(self, epoch, logs=None):
        logs['lr'] = self.lr
        logs['lr_period_no'] = self.period_no
        logs['lr_batches_passed'] = self.batches_passed_since_restart
        logs['lr_period_end'] = 0
        if self.batches_passed_since_restart >= self.batches_period:
            logs['lr_period_end'] = 1
            filename = ''
            if self.filepath:
                filename = self.filepath + '.' + str(self.period_no) + '.model'
                if self.save_weights_only:
                    self.model.save_weights(filename, overwrite=True)
                else:
                    self.model.save(filename, overwrite=True)
            if self.verbose:
                print('Period {periiod} completed. Model saved at {filename}. Stats: {stats}'.format(
                    periiod= self.period_no, filename = filename, stats = logs
                ))

            self.batches_passed_since_restart = 0
            self.period_no += 1

if __name__== "__main__":
    params = {
        'seed': 241075,
        'model': 'FNN',
        'backbone': 'resnet34',
        'initial_weightns': 'imagenet',
        'optimizer': 'sgd',
        'optimizer_params': {},
        'augmented_image_size': 128,
        'padded_image_size': 192,
        'nn_image_size': 128,
        'channels': 3,
        'coord_conv': False,  # default False. CoordinatesConv layers (R and B channels)
        'norm_sigma_k': 1.,

        'load_model_from': None,  # 'models_1/FNN_resnet34_adam_960x544_f1_9e2580.model',

        'train_augmentation_mode': 'basic',
        'test_augmentation_mode': 'inference',

        'epochs_warmup': 2,
        'epochs': 102,
        'batch_size': 20,
        'test_batch_size': 50,

        'cosine_annealing_params' : {'min_lr' : 2.e-5, 'max_lr': 1.e-2, 'period' : 10, 'verbose' : 1},

        'monitor_metric': ('val_my_iou_metric', 'max'),  # default 'val_my_iou_metric'

        'ReduceLROnPlateau': {
            'factor': 0.5,
            'patience': 10,
            'min_lr': 1e-6,
        },
        'EarlyStopping': {'patience': 50},

        'test_fold_no': 1,

        'attempt': 1,
        'comment': '',
    }
    params = type("params", (object,), params)
    from run_seg_test import RunTest
    RunTest(params)