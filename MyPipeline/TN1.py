
# coding: utf-8

# # Загрузка тестовых данных

# In[1]:


import sys
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm_notebook, tnrange

import torch


# In[2]:


DEV_MODE = False

basicpath = 'T:/Kaggle_Data/Salt/'
path_train = basicpath + 'train/'
path_test = basicpath + 'test/'


# In[3]:


depths_df = pd.read_csv(basicpath+"/depths.csv", index_col="id")
train_df = pd.read_csv(basicpath+"/train.csv", index_col="id", usecols=[0])
train_df = train_df.join(depths_df)
test_df = depths_df[~depths_df.index.isin(train_df.index)]

folds_df = pd.read_csv(basicpath+"/test_folds.csv", index_col="id")
train_df = folds_df.join(train_df)


# In[4]:


if DEV_MODE:
    train_df = train_df.head(100)
    test_df = test_df.head(200)
    depths_df = depths_df[depths_df.index.isin(train_df.index) | depths_df.index.isin(test_df.index)]
print(train_df.shape, test_df.shape, depths_df.shape)


# In[5]:


def load_image(path, mask = False):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)
    
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if mask:
        # Convert mask to 0 and 1 format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = torch.from_numpy(img // 255)
        return img.float()
    else:
        img = torch.from_numpy(img / 255.0)
        return img
        #return img.float().reshape((img.shape[0],img.shape[1],1)).permute([2, 0, 1])


# In[6]:


def LoadImages(df, train_data = True):
    path = path_train if train_data else path_test
    path_images = path + 'images/'
    path_masks  = path + 'masks/'
    df["images"] = [np.array(load_image(path_images+"{}.png".format(idx))) for idx in tqdm_notebook(df.index)]
    if train_data:
        df["masks"] = [np.array(load_image(path_masks+"{}.png".format(idx), mask=True)) for idx in tqdm_notebook(df.index)]
    


# In[7]:


LoadImages(train_df)
img_size_ori = train_df['images'][0].shape[1]


# # Datasets

# In[8]:


test_fold_no = 0

train_images = train_df.images[train_df.test_fold != test_fold_no]
train_masks  = train_df.masks[train_df.test_fold != test_fold_no]
validate_images = train_df.images[train_df.test_fold == test_fold_no]
validate_masks  = train_df.masks[train_df.test_fold == test_fold_no]


# In[9]:


train_masks[0].shape


# # Dataset and augmentation

# In[10]:


mean_val = np.mean(train_images.apply(np.mean))
mean_std = np.mean(train_images.apply(np.std))
mean_val, mean_std 


# In[11]:


train_df.shape


# In[12]:


nn_image_size = 96


# In[13]:


sys.path.insert(1, '../3rd_party/albumentations')
sys.path.insert(1, '../3rd_party/imgaug')
import albumentations


# In[14]:


from torch.utils import data


# In[15]:


def basic_aug(p=1.):
    return albumentations.Compose([
        albumentations.HorizontalFlip(),
        albumentations.RandomCrop(nn_image_size, nn_image_size),
        albumentations.Normalize(mean = mean_val, std = mean_std, max_pixel_value = 1.0),
    ], p=p)
augmentation = basic_aug()


# In[16]:


class TGSSaltDataset(data.Dataset):
    def __init__(self, images, masks = None):
        self.images = images
        self.masks = masks
        self.num_inputs  =1
        self.num_targets = 1
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        assert index in range(0, self.__len__())
        
        image = self.images[index]
        mask = self.masks[index] if not self.masks is None else None
        aug_res = augmentation(image = image, mask = mask)
        image = aug_res['image']
        image = torch.from_numpy(image).float().permute([2, 0, 1])
        if not self.masks is None:
            mask = torch.from_numpy(aug_res['mask']).float().reshape((1, image.shape[1],image.shape[2]))
            return (image, mask,)
        else:
            return (image,)


# # IOU loss

# In[17]:


thresholds = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

def iou(img_true, img_pred):
    i = np.sum(((img_true*img_pred) >0))
    u = np.sum(((img_true + img_pred) >0))
    if u == 0:
        return u
    return i/u

def iou_metric(imgs_true, imgs_pred):
    num_images = len(imgs_true)
    scores = np.zeros(num_images)
    
    for i in range(num_images):
        if imgs_true[i].sum() == imgs_pred[i].sum() == 0:
            scores[i] = 1.
        else:
            scores[i] = (thresholds <= iou(imgs_true[i], imgs_pred[i])).mean()
            
    return scores.mean()


# In[18]:


def iou_metric_batch(y_true_in, y_pred_in):
    y_pred_in = (y_pred_in > 0.5).cpu().numpy() # added by sgx 20180728
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch].cpu().numpy(), y_pred_in[batch])
        metric.append(value)
    #print("metric = ",metric)
    return np.mean(metric)


# # Модель

# In[19]:


sys.path.append(r'../3rd_party/pytorch-summary')
import torchsummary


# In[20]:


sys.path.insert(1, '../3rd_party/TernausNetV2')
from models.ternausnet2 import TernausNetV2


# In[ ]:


def get_model(model_path):
    model = TernausNetV2(num_classes=1, num_filters=32, num_input_channels=1)
    state = torch.load('../TernausNetV2/weights/deepglobe_buildings.pt')
    state = {key.replace('module.', '').replace('bn.', ''): value for key, value in state['model'].items()}

    #model.load_state_dict(state)
    model.train()

    if torch.cuda.is_available():
        model.cuda()
    return model


# In[ ]:


model = get_model('weights/deepglobe_buildings.pt')


# In[ ]:


#print(model)


# In[ ]:


torchsummary.summary(model, (1, 96, 96))


# In[24]:


from torchvision import models
model = models.vgg11().features
model.cuda()


# In[25]:


torchsummary.summary(model, (3, 96, 96))


# In[51]:


import model_0
from imp import reload
model_0 = reload(model_0)
model = model_0.get_model().cuda()


# In[39]:


torchsummary.summary(model, (3, 96, 96))


# # Обучение (torchtools)

# In[40]:


sys.path.append('../3rd_party/torchtools')
sys.path.append('../3rd_party/tensorboard_logger')
sys.path.append('../3rd_party/tensorboardX')
import torchtools.trainer
import imp
torchtools.trainer = imp.reload(torchtools.trainer)
from torchtools.meters import LossMeter, AccuracyMeter
from torchtools.callbacks import (
    StepLR, ReduceLROnPlateau, TensorBoardLogger, CSVLogger)


# In[34]:


dataset = TGSSaltDataset(train_images, train_masks)
train_data_loader = data.DataLoader(dataset, batch_size = 30, shuffle = True)
dataset_val = TGSSaltDataset(validate_images, validate_masks)
val_data_loader = data.DataLoader(dataset_val, batch_size = 50, shuffle = False)

learning_rate = 1e-4
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

EPOCHS = 20

trainer = torchtools.trainer.Trainer(model, train_data_loader, criterion, optimizer, val_data_loader, device='cuda')

# Callbacks

loss = LossMeter('loss')
val_loss = LossMeter('val_loss')
acc = AccuracyMeter('acc')
val_acc = AccuracyMeter('val_acc')
scheduler = StepLR(optimizer, 1, gamma=0.95)
reduce_lr = ReduceLROnPlateau(optimizer, 'val_loss', factor=0.3, patience=3)
logger = TensorBoardLogger()
csv_logger = CSVLogger(keys=['epochs', 'loss', 'acc', 'val_loss', 'val_acc'])

trainer.register_hooks([
    loss, val_loss, acc, val_acc, scheduler, reduce_lr, logger, csv_logger])

_ = trainer.train(EPOCHS)


# # Обучение (torchsample)

# In[41]:


sys.path.append('../3rd_party/torchsample')
sys.path.append('../3rd_party/nibabel')
from torchsample.modules import ModuleTrainer
from torchsample.callbacks import EarlyStopping, ReduceLROnPlateau
from torchsample.regularizers import L1Regularizer, L2Regularizer
from torchsample.constraints import UnitNorm
from torchsample.initializers import XavierUniform
from torchsample.metrics import Metric, BinaryAccuracy
from torchsample import TensorDataset


# In[42]:


from fnmatch import fnmatch


# In[43]:


class MyIouMetric(Metric):

    def __init__(self):
        self.total = 0
        self.total_count = 0
        self._name = 'my_iou_metric'

    def reset(self):
        self.total = 0
        self.total_count = 0

    def __call__(self, y_pred, y_true):
        self.total += iou_metric_batch(y_true, y_pred)
        self.total_count += 1
        return self.total/self.total_count


# In[46]:


train_loader = data.DataLoader(TGSSaltDataset(train_images, train_masks), batch_size = 25, shuffle = True)
val_loader = data.DataLoader(TGSSaltDataset(validate_images, validate_masks), batch_size = 50, shuffle = False)

learning_rate = 1e-4
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainer = ModuleTrainer(model)

callbacks = [EarlyStopping(patience=30),
             ReduceLROnPlateau(factor=0.5, patience=10)]
regularizers = [L1Regularizer(scale=1e-3, module_filter='*'),
                L2Regularizer(scale=1e-5, module_filter='*')]
constraints = [UnitNorm(frequency=3, unit='batch', module_filter='*')]
initializers = [XavierUniform(bias=False, module_filter='*')]
metrics = [MyIouMetric()]

trainer.compile(loss=loss_fn,
                optimizer=optimizer,
                regularizers=None, #regularizers,
                constraints=None,#constraints,
                initializers=None,#initializers,
                metrics=metrics,#metrics, 
                callbacks=callbacks)

trainer.fit_loader(train_loader, val_loader, num_epoch=20, cuda_device=0, verbose=1)


# # Обучение (ingite)

# In[49]:


sys.path.append('../3rd_party/ignite')
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import BinaryAccuracy, Loss


# In[ ]:


train_loader = data.DataLoader(TGSSaltDataset(train_images, train_masks), batch_size = 25, shuffle = True)
val_loader = data.DataLoader(TGSSaltDataset(validate_images, validate_masks), batch_size = 50, shuffle = False)

learning_rate = 1e-4
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainer = create_supervised_trainer(model, optimizer, loss_fn, device = "cuda")
evaluator = create_supervised_evaluator(model, device = "cuda",
                                        metrics={
                                            'accuracy': BinaryAccuracy(),
                                            'my_loss': Loss(loss_fn)
                                        })

@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(trainer):
    #print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))
    pass

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['accuracy'], metrics['accuracy']))

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['accuracy'], metrics['accuracy']))

trainer.run(train_loader, max_epochs=20)


# In[45]:


import tqdm

dataset = TGSSaltDataset(train_images, train_masks)
dataset_val = TGSSaltDataset(validate_images, validate_masks)

learning_rate = 1e-4
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for e in range(100):
    train_loss = []
    train_iou = []
    for image, mask in tqdm.tqdm_notebook (data.DataLoader(dataset, batch_size = 30, shuffle = True)):
        image = image.type(torch.float).cuda()
        y_pred = model(image)
        loss = loss_fn(y_pred, mask.cuda())

        print(type(y_pred.cpu()), type(mask.cpu()))
        
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        train_loss.append(loss.item())
        #train_iou.append
        #(iou_metric_batch(mask.cpu(), y_pred.cpu()))
        
    val_loss = []
    test_iou = []
    for image, mask in data.DataLoader(dataset_val, batch_size = 50, shuffle = False):
        image = image.cuda()
        y_pred = model(image)
        loss = loss_fn(y_pred, mask.cuda())
        val_loss.append(loss.item())
        #test_iou.append(iou_metric_batch(mask.cpu(), y_pred.cpu()))

    with open(r'D:\Temp\log.txt', 'w') as f:
        print("Epoch: %d, Train: %.3f, Val: %.3f" % (e, np.mean(train_loss), np.mean(val_loss)), file = f)        
    print("Epoch: %d, Train: %.3f, Val: %.3f" % (e, np.mean(train_loss), np.mean(val_loss)))

