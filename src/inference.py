#!/usr/bin/env python 
# coding: utf-8 

# # torch.save(model.state_dict(), '/Data/ndionelis/formodels/sgfrmr2ehwh')        
# # data_dir = '/Data/ndionelis/bingmap/maindatasetnewnew2' 
# # Potsdam       
# # data_dir = '/Data/ndionelis/mainpo2'
# # On the dataset Vaihingen       
# # data_dir = '/Data/ndionelis/mainva2'

# # This is to test: torch.save(model.state_dict(), './segformermain30082023')         
# # Also: data_dir = '../../CVUSA/bingmap/mainTheDataset' 

# # pip install pytorch_ssim                    
    
import sys  
#get_ipython().system('{sys.executable} -m pip install pytorch_ssim') 
#get_ipython().system('{sys.executable} -m pip install pytorch_msssim')  
 
# #!{sys.executable} -m conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# #!{sys.executable} -m pip install timm   

# import numpy as np 
# from skimage import io 
# from glob import glob
# from tqdm import tqdm_notebook as tqdm
# from sklearn.metrics import confusion_matrix
# import random
# import itertools
# # # Matplotlib           
# import matplotlib.pyplot as plt
# #get_ipython().run_line_magic('matplotlib', 'inline')               
# #from IPython import get_ipython    
# #get_ipython().run_line_magic('matplotlib', 'inline')  
# #exec(%matplotlib inline)
# # # Torch imports     
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data as data
# import torch.optim as optim
# import torch.optim.lr_scheduler
# import torch.nn.init
# from torch.autograd import Variable
# import torchvision.transforms as T
# import albumentations as A
# import segmentation_models_pytorch as smp
# import kornia

# WINDOW_SIZE = (256, 256) # Patch size  
# STRIDE = 32 # # # # Stride for testing 
# IN_CHANNELS = 3 # Number of input channels (e.g. RGB)  
# #FOLDER = "./ISPRS_dataset/" # # Replace with your "/path/to/the/ISPRS/dataset/folder/"
# #FOLDER = "../" 
# FOLDER = "/Data/ndionelis/"
# #BATCH_SIZE = 10 # Number of samples in a mini-batch     
# #BATCH_SIZE = 64  
# #BATCH_SIZE = 10 
# BATCH_SIZE = 10

# LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # # Label names
# N_CLASSES = len(LABELS) # # Number of classes 
# #print(N_CLASSES)  

# WEIGHTS = torch.ones(N_CLASSES) # # # Weights for class balancing 
# CACHE = True # # Store the dataset in-memory  

# #DATASET = 'Vaihingen'    
# DATASET = 'Potsdam'    

# DATASET2 = 'Vaihingen'

# if DATASET == 'Potsdam':
#     MAIN_FOLDER = FOLDER + 'Potsdam/'
#     # Uncomment the next line for IRRG data      
#     # DATA_FOLDER = MAIN_FOLDER + '3_Ortho_IRRG/top_potsdam_{}_IRRG.tif'     
#     # # For RGB data    
#     #print(MAIN_FOLDER)   
#     #sadfszf  
#     #print(MAIN_FOLDER) 
#     #asdfklsz 
#     DATA_FOLDER = MAIN_FOLDER + '2_Ortho_RGB/top_potsdam_{}_RGB.tif'
#     LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
#     ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'    

# elif DATASET == 'Vaihingen':
#     MAIN_FOLDER = FOLDER + 'Vaihingen/'
#     #print(MAIN_FOLDER)      
#     #asdfszdf      
#     DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
#     LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
#     ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'

# # imports and stuff
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
# # Matplotlib  
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline') 
# # Torch imports   
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
import torchvision.transforms as T
import albumentations as A
import segmentation_models_pytorch as smp
import kornia

import time   

WINDOW_SIZE = (220, 220) # Patch size 
#WINDOW_SIZE = (512, 512) # Patch size 
STRIDE = 32 # # # # # Stride for testing 
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
#FOLDER = "./ISPRS_dataset/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
#FOLDER = "../../"
#FOLDER = "/Data/ndionelis/"
FOLDER = '../' 
#BATCH_SIZE = 10 # # Number of samples in a mini-batch 
#BATCH_SIZE = 64  
#BATCH_SIZE = 128    
#BATCH_SIZE = 256 
BATCH_SIZE = 10  
#BATCH_SIZE = 30 

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # # Label names 
N_CLASSES = len(LABELS) # # Number of classes   
#print(N_CLASSES)   

WEIGHTS = torch.ones(N_CLASSES) # # # Weights for class balancing
CACHE = True # # Store the dataset in-memory 

DATASET = 'Vaihingen' 
#DATASET = 'Potsdam'

if DATASET == 'Potsdam':
    MAIN_FOLDER = FOLDER + 'Potsdam/'
    # Uncomment the next line for IRRG data
    # DATA_FOLDER = MAIN_FOLDER + '3_Ortho_IRRG/top_potsdam_{}_IRRG.tif'
    # For RGB data
    DATA_FOLDER = MAIN_FOLDER + '2_Ortho_RGB/top_potsdam_{}_RGB.tif'
    LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
    ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'    
elif DATASET == 'Vaihingen':
    MAIN_FOLDER = FOLDER + 'Vaihingen/'
    DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'

palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)
invert_palette = {v: k for k, v in palette.items()}

def ade_palette():  
    """ADE20K palette that maps each class to RGB values. """
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

def convert_to_color(arr_2d, palette=palette):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d

# img = io.imread('/Data/ndionelis/Vaihingen/top/top_mosaic_09cm_area11.tif')
# fig = plt.figure()
# fig.add_subplot(121)
# plt.imshow(img)

# gt = io.imread('/Data/ndionelis/Vaihingen/gts_for_participants/top_mosaic_09cm_area11.tif')
# fig.add_subplot(122) 
# plt.imshow(gt)
# plt.show()
 
# array_gt = convert_from_color(gt)
# print("Ground truth in numerical format has shape ({},{}) : \n".format(*array_gt.shape[:2]), array_gt)

def get_random_pos(img, window_shape):
    w, h = window_shape
    W, H = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

def CrossEntropy2d(input, target, weight=None, size_average=True):
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0),input.size(1), -1)
        output = torch.transpose(output,1,2).contiguous()
        output = output.view(-1,output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target,weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

def sliding_window(top, step=10, window_size=(20,20)):
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]
            
def count_sliding_window(top, step=10, window_size=(20,20)):
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(
           gts,
           predictions,
           range(len(label_values))) 
    
    print("Confusion matrix :")
    print(cm)
    
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))

    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe);
    print("Kappa: " + str(kappa))
    
    return accuracy 

# The Dataset class
class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                            cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()
        self.augmentation = augmentation
        self.cache = cache
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))
        self.data_cache_ = {}
        self.label_cache_ = {}
            
    def __len__(self):
        return 10000
    
    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        #will_rotate = False
        #will_rotate2 = False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            
            results.append(np.copy(array))  
            
        return tuple(results)
    
    def __getitem__(self, i):
        random_idx = random.randint(0, len(self.data_files) - 1)
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            data = 1/255 * np.asarray(io.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data
            
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else: 
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]
        
        data_p, label_p = self.data_augmentation(data_p, label_p)

        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))


from torch.autograd import Function
#import torch.autograd

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class SegNet(nn.Module):
    # # SegNet network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    
    def __init__(self, in_channels=IN_CHANNELS, out_channels=N_CLASSES):
        super(SegNet, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2)
        
        self.conv1_1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(64)
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(128)
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(256)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_bn = nn.BatchNorm2d(512)
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_bn = nn.BatchNorm2d(512)
        
        #self.ll00 = nn.Linear(1024, 2)  
        self.ll00 = nn.Linear(512, 3)
        
        self.conv5_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3_D_bn = nn.BatchNorm2d(512)
        self.conv5_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(512)
        self.conv5_1_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(512)
        
        self.conv4_3_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3_D_bn = nn.BatchNorm2d(512)
        self.conv4_2_D = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_2_D_bn = nn.BatchNorm2d(512)
        self.conv4_1_D = nn.Conv2d(512, 256, 3, padding=1)
        self.conv4_1_D_bn = nn.BatchNorm2d(256)
        
        self.conv3_3_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3_D_bn = nn.BatchNorm2d(256)
        self.conv3_2_D = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_2_D_bn = nn.BatchNorm2d(256)
        self.conv3_1_D = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3_1_D_bn = nn.BatchNorm2d(128)
        
        self.conv2_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(128)
        self.conv2_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(64)
        
        self.conv1_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(64)
        self.conv1_1_D = nn.Conv2d(64, out_channels, 3, padding=1)
        
        self.apply(self.weight_init)
        
    def forward(self, x, alpha):
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)
        
        # Encoder block 5 
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)
        
        #print(x) 
        #print(x.shape)
        
        #xx22 = F.adaptive_avg_pool2d(x, 1).reshape(BATCH_SIZE, -1) 
        #xx22 = F.adaptive_avg_pool2d(x, 1).reshape(2*BATCH_SIZE, -1)
        xx22 = F.adaptive_avg_pool2d(x, 1).reshape(BATCH_SIZE, -1)
        
        #xx22 = self.ll00(xx22) 
        
        xx22 = ReverseLayerF.apply(xx22, alpha)
    
        xx22 = self.ll00(xx22)
        
        xx22 = F.softmax(xx22)
        
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))
        
        # Decoder block 4  
        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3 
        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # # # Decoder block 2  
        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1 
        x = self.unpool(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = F.log_softmax(self.conv1_1_D(x))
        #return x 
        return x, xx22
    
    def forward2(self, x):
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x, mask1 = self.pool(x)
        
        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x, mask2 = self.pool(x)
        
        # Encoder block 3
        x = self.conv3_1_bn(F.relu(self.conv3_1(x)))
        x = self.conv3_2_bn(F.relu(self.conv3_2(x)))
        x = self.conv3_3_bn(F.relu(self.conv3_3(x)))
        x, mask3 = self.pool(x)
        
        # Encoder block 4
        x = self.conv4_1_bn(F.relu(self.conv4_1(x)))
        x = self.conv4_2_bn(F.relu(self.conv4_2(x)))
        x = self.conv4_3_bn(F.relu(self.conv4_3(x)))
        x, mask4 = self.pool(x)
        
        # Encoder block 5 
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x = self.conv5_3_bn(F.relu(self.conv5_3(x)))
        x, mask5 = self.pool(x)
        
        #print(x) 
        #print(x.shape)
        
        #xx22 = F.adaptive_avg_pool2d(x, 1).reshape(BATCH_SIZE, -1) 
        #xx22 = F.adaptive_avg_pool2d(x, 1).reshape(2*BATCH_SIZE, -1)
        #xx22 = F.adaptive_avg_pool2d(x, 1).reshape(BATCH_SIZE, -1)
        
        #xx22 = self.ll00(xx22) 
        
        #xx22 = ReverseLayerF.apply(xx22, alpha)
    
        #xx22 = self.ll00(xx22)
        
        #xx22 = F.softmax(xx22)
        
        x = self.unpool(x, mask5)
        x = self.conv5_3_D_bn(F.relu(self.conv5_3_D(x)))
        x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))
        
        # Decoder block 4  
        x = self.unpool(x, mask4)
        x = self.conv4_3_D_bn(F.relu(self.conv4_3_D(x)))
        x = self.conv4_2_D_bn(F.relu(self.conv4_2_D(x)))
        x = self.conv4_1_D_bn(F.relu(self.conv4_1_D(x)))
        
        # Decoder block 3 
        x = self.unpool(x, mask3)
        x = self.conv3_3_D_bn(F.relu(self.conv3_3_D(x)))
        x = self.conv3_2_D_bn(F.relu(self.conv3_2_D(x)))
        x = self.conv3_1_D_bn(F.relu(self.conv3_1_D(x)))
        
        # # # Decoder block 2   
        x = self.unpool(x, mask2)
        x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))
        
        # Decoder block 1  
        x = self.unpool(x, mask1)
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = F.log_softmax(self.conv1_1_D(x))
        return x 
        #return x, xx22 

net = SegNet()
 
import os
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

# # # Download VGG-16 weights from PyTorch 
vgg_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
if not os.path.isfile('./vgg16_bn-6c64b313.pth'):
    weights = URLopener().retrieve(vgg_url, './vgg16_bn-6c64b313.pth')

vgg16_weights = torch.load('./vgg16_bn-6c64b313.pth')
mapped_weights = {}
for k_vgg, k_segnet in zip(vgg16_weights.keys(), net.state_dict().keys()):
    if "features" in k_vgg:
        mapped_weights[k_segnet] = vgg16_weights[k_vgg]
        print("Mapping {} to {}".format(k_vgg, k_segnet))
        
try:
    net.load_state_dict(mapped_weights)
    print("Loaded VGG-16 weights in SegNet !")
except:
    # Ignore missing keys 
    #pass
    pass

#net.cuda() 

# The model SegFormer 

# # This is to test: torch.save(model.state_dict(), './segformermain30082023')
# # Also: data_dir = '../../CVUSA/bingmap/mainTheDataset'

from transformers import SegformerForSemanticSegmentation
#import transformers 

# id2label = {0 : 'Impervious surfaces',
#             1 : 'Buildings',
#             2 : 'Low vegetation',
#             3 : 'Trees',
#             4 : 'Cars',
#             5 : 'Clutter'}
# label2id = {v: k for k, v in id2label.items()} 

#from transformers import SegformerForSemanticSegmentation  
import json
from huggingface_hub import cached_download, hf_hub_url

# # load id2label mapping from a JSON on the hub       
#repo_id = "datasets/huggingface/label-files" 
#filename = "./ade20k-id2label.json" 
filename = "./ade20kid2label.json"
#id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename)), "r")) 
id2label = json.load(open(filename, "r")) 
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

net = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                         num_labels=6, 
                                                         id2label=id2label, 
                                                         label2id=label2id,
                                                         alpha=0.0,
)           

# net = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
#                                                          num_labels=19, 
#                                                          id2label=id2label, 
#                                                          label2id=label2id,
#                                                          alpha=0.0,
# )

net.cuda()

# #base_lr = 0.01 
# #base_lr = 0.04
# #base_lr = 0.08  
# base_lr = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CLASSES = 6    
#print(N_CLASSES)            

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # # # # # # /home/nikolaos/CVUSA/bingmap  

    # #print(test_images)      
    # #print(test_images.size)  
    
    all_preds = []
    all_gts = []
    
    # # # Switch the network to inference mode       
    #net.eval()
    
    net.eval()

    for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False): 
        #img = (1 / 255 * np.asarray(io.imread('/home/nikolaos/CVUSA/bingmap/0000001.jpg'), dtype='float32'))                  
        
        #img = (1 / 255 * np.asarray(io.imread('/home/nikolaos/CVUSA/bingmap/19/'+str(44514).zfill(7)+'.jpg'), dtype='float32'))
        
        # data_dir = '/Data/ndionelis/inputtss'   

        countertotal = 0
        foraverage = 0
        foraverage2 = 0 
        foraverage3 = 0 
        
        #data_dir = '/Data/ndionelis/bingmap/main_dataset/val/1' 

        #data_dir = '/Data/ndionelis/bingmap/maindatasetnewnew/val/1'     

        #data_dir = '../mubsa54b2bc2h/val/1'           
        #data_dir = '../mubsa54b2bc2a/val/1'  
        #data_dir = '/home/ndionelis/mubsa54b2bc2a/val/1'   
        #data_dir = '/Data/ndionelis/1'  
        data_dir = '/Data/ndionelis/SegmentationNew/Inputs/InputImages'
        #data_dir = '/Data/ndionelis/Folder_CVUSA_Segmentation/Inputs/Inputs'   
        
        # # torch.save(model.state_dict(), '/Data/ndionelis/formodels/sgfrmr2ehwh')                                                  
        # # data_dir = '/Data/ndionelis/bingmap/maindatasetnewnew2'           
        # # # Potsdam               
        # # data_dir = '/Data/ndionelis/mainpo2'    
        # # On the dataset Vaihingen           
        # # data_dir = '/Data/ndionelis/mainva2'

        # # This is to test: torch.save(model.state_dict(), './segformermain30082023')                        
        # # Also: data_dir = '../../CVUSA/bingmap/mainTheDataset'        
        
        for file in os.listdir(data_dir):                   
            img = (1 / 255 * np.asarray(io.imread(os.path.join(data_dir, file)), dtype='float32'))

            countertotal += 1
            counterr11 = file

            print(counterr11)
            print(countertotal) 

            #img = (1 / 255 * np.asarray(io.imread('/home/nikolaos/CVUSA/bingmap/19/'+str(44515).zfill(7)+'.jpg'), dtype='float32'))

            pred = np.zeros(img.shape[:2] + (N_CLASSES,)) 

            #img = (1 / 255 * np.asarray(io.imread('/home/nikolaos/CVUSA/bingmap/19/0000019.jpg'), dtype='float32'))                                                                   

            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False)):
                # # # # # Display in progress results     
                if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                        _pred = np.argmax(pred, axis=-1)
                        #                     fig = plt.figure()
                        #                     #fig.add_subplot(1,3,1)
                        #                     fig.add_subplot(1,2,1)
                        #                     plt.imshow(np.asarray(255 * img, dtype='uint8'))
                        #                     plt.axis('off')
                        #                     #fig.add_subplot(1,3,2)
                        #                     fig.add_subplot(1,2,2)
                        #                     plt.imshow(convert_to_color(_pred))
                        #                     plt.axis('off')
                        #                     #fig.add_subplot(1,3,3)
                        #                     #plt.imshow(gt)
                        #                     #plt.axis('off') 
                        #                     #clear_output()
                        #                     plt.show()

                # # # # # Build the tensor     
                #image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords] 
                import copy
                image_patches = [copy.deepcopy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
                
                image_patches = np.asarray(image_patches)
                #image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
                with torch.no_grad():
                    #image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)      
                    image_patches = Variable(torch.from_numpy(image_patches).cuda())
                    
                    # # /home/nikolaos/CVUSA/bingmsap              
                    
                    # # Do the inference           
                    #outs = net(image_patches)   
                    #outs, _ = net(image_patches, 0.01) 
                    #outs = net.forward2(image_patches)
                    #outs = outs.data.cpu().numpy() 
                    
                    # # prepare the image for the model 
                    #encoding = feature_extractor(image_patches, return_tensors="pt")
                    #pixel_values = encoding.pixel_values.to(device)

                    #encoding = feature_extractor(image_patches, return_tensors="pt") 
                    #pixel_values = encoding.pixel_values.to(device)
                    
                    pixel_values = image_patches.to(device) 

                    #outputs = model(pixel_values=pixel_values) 

                    outputs = net(pixel_values=pixel_values) 

                    logits = outputs.logits.cpu()
                    upsampled_logits = nn.functional.interpolate(logits,
                                    size=image_patches.shape[3], # # (height, width)   
                                    mode='bilinear',
                                    align_corners=False)
                    #seg = upsampled_logits.argmax(dim=1)[0] 
                    outs = upsampled_logits.detach().cpu().numpy()

                    # # /home/nikolaos/CVUSA/bingmsap             

                    # # # # # Do the inference               
                    #outs = net(image_patches)    
                    #outs, _ = net(image_patches, 0.01)
                    #outs = net.forward2(image_patches)
                    
                    #outs = net.forward2(image_patches)
                    
                    # For the Transformer                          
                    #outs = outs.data.cpu().numpy()      

                    # # # Fill in the results array      
                    for out, (x, y, w, h) in zip(outs, coords):
                        out = out.transpose((1,2,0))
                        pred[x:x+w, y:y+h] += out
                    del(outs)

                    #break

            pred = np.argmax(pred, axis=-1)  

            #import pdb; pdb.set_trace() 

            from samgeo.hq_sam import SamGeo, show_image, download_file, overlay_images, tms_to_geotiff

            sam_kwargs = {
                "points_per_side": 32,
                "pred_iou_thresh": 0.86,
                "stability_score_thresh": 0.92,
                "crop_n_layers": 1,
                "crop_n_points_downscale_factor": 2,
                "min_mask_region_area": 100,
            }    

            sam = SamGeo(
                model_type="vit_h",
                checkpoint="sam_vit_h_4b8939.pth",
                sam_kwargs=sam_kwargs,
            )          

            import cv2
            import matplotlib.pyplot as plt
            
            plt.figure()
            #plt.imshow(pixel_values[mainvarloop,:,:,:].permute(1, 2, 0).cpu().numpy())     
            plt.imshow(np.asarray(255 * img, dtype='uint8'))
            #img = (1 / 255 * np.asarray(io.imread(os.path.join(data_dir, '0044478.jpg')), dtype='float32'))
            #plt.imshow(np.asarray(255 * img, dtype='uint8'))
            plt.axis('off') 

            # sam_eo = SamEO(checkpoint="sam_vit_h_4b8939.pth",
            #    model_type='vit_h',
            #    device=device,
            #    erosion_kernel=(3, 3),
            #    mask_multiplier=255,
            #    sam_kwargs=None)
            
            # pred_tiff_path = 'pred.tiff' 
            
            # #sam_eo.tiff_to_tiff(tms_tiff_path, pred_tiff_path)  
            # sam_eo.tiff_to_tiff("SAMGeoSAMHQInp5B.png", pred_tiff_path)

            # pred_image = cv2.cvtColor(cv2.imread(pred_tiff_path), cv2.COLOR_BGR2RGB) 

            # plt.figure() 
            # f, axarr = plt.subplots(1,2) 
            # axarr[0].imshow(image)
            # axarr[1].imshow(pred_image)
            # plt.show()
            
            #from torchvision.transforms import functional as F                
            #F.to_pil_image(image_tensor)   
            #sam.generate(F.to_pil_image(pixel_values[1,:,:,:].cpu()))    
            #sam.generate(F.to_pil_image(pixel_values[1,:,:,:].cpu()), output="masks2.png", foreground=True, unique=True)    
            sam.generate("SAMGeoSAMHQInp5B.png", output="masks2.png", foreground=True, unique=True) 
            #sam.generate(np.array(pixel_values[1,:,:,:].permute(1, 2, 0).cpu().numpy().astype('uint8')), output="masks2.png", foreground=True, unique=True)
            #cv2_image = numpy_image, (1, 2, 0)) 
            #cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            
            #sam.generate(cv2.cvtColor(pixel_values[1,:,:,:].permute(1, 2, 0).cpu().numpy().astype('uint8'), cv2.COLOR_BGR2RGB), output="masks2.png", foreground=True, unique=True) 
            #((np.asarray(255 * pixel_values, dtype='uint8')) / 255.0)
            #pixel_values[0,:,:,:].permute(1, 2, 0).cpu().numpy().astype('uint8')  
            #sam.generate(np.asarray(pixel_values[0,:,:,:].permute(1, 2, 0).cpu() / 255., dtype='uint8'), output="masks2.png", foreground=True, unique=True)

            #from segment_anything_hq import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor                    
            #mask_generator = SamAutomaticMaskGenerator(sam)  # The automatic mask generator    
            #masks = mask_generator.generate(pixel_values)  # # Segment the input image 
            
            #mainimage = sam.show_anns(axis="off", alpha=1, output="SAMGeoSAMHQInp5B.png")    
            
            mainimage = np.asarray(255 * (1 / 255 * np.asarray(io.imread(os.path.join('/Data/ndionelis/SegmentationNew/Outputs/OutputImages', 'output'+file[5:])), dtype='float32')), dtype='uint8')
            #mainimage = np.asarray(255 * (1 / 255 * np.asarray(io.imread(os.path.join('/Data/ndionelis/Folder_CVUSA_Segmentation/Outputs/Outputs', 'output'+file[5:])), dtype='float32')), dtype='uint8')

            #pred = mainimage                            
            #mainimage = pred    

            #mainimage = torch.zeros_like(torch.from_numpy(mainimage)).numpy()          
            #pred = torch.zeros_like(torch.from_numpy(pred)).numpy()   
            
            # # # # # Display the result                                                        
            #clear_output()          
            fig = plt.figure()
            #fig.add_subplot(1,3,1)                
            #fig.add_subplot(1,2,1)
            plt.imshow(np.asarray(255 * img, dtype='uint8'))
            plt.axis('off')
            #plt.show()  
            #fig.add_subplot(1,3,2)                                 
            #fig.add_subplot(1,2,2)   
            #print(counterr11)
            #print(countertotal)
            #counterr11 = counterr11[:-4]
            #print(counterr11)   
            #adfasdflzs    
            # try:
            #     #plt.savefig('./inputs/input'+str(counterr11)+'.png', bbox_inches='tight')        
            #     plt.savefig('./ii/input'+str(counterr11)+'.png', bbox_inches='tight')
            # except:
            #     os.mkdir('ii')
            #     plt.savefig('./ii/input'+str(counterr11)+'.png', bbox_inches='tight')     
            #plt.savefig('inputim1.png', bbox_inches='tight')                                        
            plt.savefig('/Data/ndionelis/theinput'+str(counterr11)+'.png', bbox_inches='tight')
            
            fig2 = plt.figure() 
            plt.imshow(convert_to_color(pred))
            plt.axis('off')
            #plt.show()  
            #fig.add_subplot(1,3,3)        
            #plt.imshow(gt)  
            #plt.axis('off')      
            #plt.show()  
            #plt.savefig('./tesssttt2_tttlll2.png')                                                          
            #io.imsave('./tesssttt2_tttlll2.png')         
            #plt.pause(10)     
            # try:
            #     #plt.savefig('./outputs/output'+str(counterr11)+'.png', bbox_inches='tight')       
            #     plt.savefig('./oo/output'+str(counterr11)+'.png', bbox_inches='tight')
            # except:
            #     os.mkdir('oo')
            #     plt.savefig('./oo/output'+str(counterr11)+'.png', bbox_inches='tight')    
            plt.savefig('/Data/ndionelis/theoutput'+str(counterr11)+'.png', bbox_inches='tight')
            #plt.savefig('outputim1.png', bbox_inches='tight')       

            fig2 = plt.figure()  
            plt.imshow(mainimage)        
            #plt.imshow(convert_to_color(pred))               
            plt.axis('off')  
            #plt.show()   
            #fig.add_subplot(1,3,3)                        
            #plt.imshow(gt)    
            #plt.axis('off')       
            #plt.show()  
            #plt.savefig('./tesssttt2_tttlll2.png')                                                                               
            #io.imsave('./tesssttt2_tttlll2.png')           
            #plt.pause(10)      
            # try: 
            #     #plt.savefig('./outputs/output'+str(counterr11)+'.png', bbox_inches='tight')                  
            #     plt.savefig('./oo/output'+str(counterr11)+'.png', bbox_inches='tight')  
            # except:
            #     os.mkdir('oo')
            #     plt.savefig('./oo/output'+str(counterr11)+'.png', bbox_inches='tight')      
            plt.savefig('/Data/ndionelis/thetheoutput'+str(counterr11)+'.png', bbox_inches='tight')
            #plt.savefig('outputim1.png', bbox_inches='tight')
            
            #os.chdir('/home/ndionelis/pytorch_unet/segment-anything-eo')

            #get_ipython().system('pip install rasterio')                  
            #get_ipython().system('pip install geopandas') 

            #get_ipython().system('wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')      
            #get_ipython().system('wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth') 
            #get_ipython().system('wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth')

            import cv2
            import matplotlib.pyplot as plt
            from sameo import SamEO

            # ## Initialize SemEO class          

            # # Availble SamEO arguments:           
            # checkpoint="sam_vit_h_4b8939.pth",
            # model_type='vit_h',
            # device='cpu',
            # erosion_kernel=(3, 3),
            # mask_multiplier=255,
            # sam_kwargs=None

            # Availble sam_kwargs: 
            # points_per_side: Optional[int] = 32,
            # points_per_batch: int = 64,
            # pred_iou_thresh: float = 0.88,
            # stability_score_thresh: float = 0.95,
            # stability_score_offset: float = 1.0,
            # box_nms_thresh: float = 0.7,
            # crop_n_layers: int = 0,
            # crop_nms_thresh: float = 0.7,
            # crop_overlap_ratio: float = 512 / 1500,
            # crop_n_points_downscale_factor: int = 1,
            # point_grids: Optional[List[np.ndarray]] = None,
            # min_mask_region_area: int = 0,
            # output_mode: str = "binary_mask",

            #device = 'cuda:0'      

            # sam_eo = SamEO(checkpoint="sam_vit_h_4b8939.pth",
            #             model_type='vit_h',
            #             device=device,
            #             erosion_kernel=(3, 3),
            #             mask_multiplier=255,
            #             sam_kwargs=None)   

            # sam_eo = SamEO(checkpoint="sam_vit_h_4b8939.pth",
            #             model_type='vit_h',
            #             device=device,
            #             erosion_kernel=(3, 3),
            #             mask_multiplier=255,
            #             sam_kwargs=None)     

            sam_eo = SamEO(checkpoint="/Data/ndionelis/sam_vit_h_4b8939.pth",
                        model_type='vit_h',
                        device=device,
                        erosion_kernel=(3, 3),
                        mask_multiplier=255,
                        sam_kwargs=None)

            # ## Download file from Openaerialmap and save it    
            tms_source = 'https://tiles.openaerialmap.org/642385491a8878000512126c/0/642385491a8878000512126d/{z}/{x}/{y}'
            #pt1 = (29.676840, -95.369222)
            #pt2 = (29.678559, -95.367314)
            zoom = 20
            #tms_tiff_path = 'test_tms_image.tif'    
            #tms_tiff_path = '/home/ndionelis/segmentanything/notebooks/Input2.png'  
            tms_tiff_path = '/home/ndionelis/Transformers-Tutorials/SegFormer/Inpuutt2.png'

            #image = sam_eo.download_tms_as_tiff(tms_source, pt1, pt2, zoom, tms_tiff_path)
            tiff_image = cv2.cvtColor(cv2.imread(tms_tiff_path), cv2.COLOR_BGR2RGB)

            #plt.figure() 
            #f, axarr = plt.subplots(1,2)  
            #axarr[0].imshow(image)
            #axarr[1].imshow(tiff_image)
            #plt.show()

            pred_tiff_path = 'pred.tiff' 

            #tms_tiff_path = 'test_tms_image.tif'  
            #tms_tiff_path = '/home/ndionelis/segmentanything/notebooks/Input1.tif'         
            #tms_tiff_path = '/home/ndionelis/segmentanything/notebooks/Input3.png'
            #tms_tiff_path = '../../../segmentanything/notebooks/Input3.png' 
            #tms_tiff_path = '../segmentanything/notebooks/Inpu5.png'   
            #tms_tiff_path = '/home/ndionelis/segmentanything/notebooks/Inpu2.png'  
            sam_eo.tiff_to_tiff('/Data/ndionelis/theinput'+str(counterr11)+'.png', pred_tiff_path)
            #sam_eo.tiff_to_tiff(tms_tiff_path, pred_tiff_path)

            pred_image = cv2.cvtColor(cv2.imread(pred_tiff_path), cv2.COLOR_BGR2RGB)

            plt.figure() 
            #f, axarr = plt.plot()     
            #f, axarr = plt.subplots(1,2)   
            #axarr[0].imshow(image)  
            #axarr[1].imshow(pred_image)   
            #plt.imshow(cv2.imread(tms_tiff_path))   
            #plt.imshow(pred_image, alpha=0.5)  
            plt.imshow(pred_image) 
            #plt.show()  
            plt.axis('off')   

            #plt.savefig('SAMGeospatial5.png')      
            #plt.savefig('/home/ndionelis/segment-anything-eo/SAMGeospatial5.png', bbox_inches='tight')   
            plt.savefig('/Data/ndionelis/SAMGeospatial5.png', bbox_inches='tight')

            sam_eo.tiff_to_tiff('/Data/ndionelis/thetheoutput'+str(counterr11)+'.png', pred_tiff_path)
            #sam_eo.tiff_to_tiff(tms_tiff_path, pred_tiff_path) 

            pred_image2 = pred_image
            
            pred_image = cv2.cvtColor(cv2.imread(pred_tiff_path), cv2.COLOR_BGR2RGB)

            plt.figure() 
            #f, axarr = plt.plot()     
            #f, axarr = plt.subplots(1,2)         
            #axarr[0].imshow(image)     
            #axarr[1].imshow(pred_image)    
            #plt.imshow(cv2.imread(tms_tiff_path))     
            #plt.imshow(pred_image, alpha=0.5)   
            plt.imshow(pred_image)  
            #plt.show()  
            plt.axis('off')   

            #plt.savefig('SAMGeospatial5.png')        
            #plt.savefig('/home/ndionelis/segment-anything-eo/SAMGeospatial5.png', bbox_inches='tight')            
            plt.savefig('/Data/ndionelis/SAMSAMGeospatial5.png', bbox_inches='tight') 

            #pred_image = np.resize(pred_image, np.shape(pred_image2))             
            pred_image2 = np.resize(pred_image2, np.shape(pred_image))

            #criterion = nn.MSELoss()       
            #loss = torch.sqrt(criterion(pred_image2, pred_image))         
            #loss = torch.sqrt(criterion(torch.from_numpy(pred_image).float(), torch.from_numpy(pred_image2).float()))        
            loss = torch.abs(torch.mean((torch.from_numpy(pred_image).float() - torch.from_numpy(pred_image2).float())))  
            loss /= torch.abs(torch.mean((torch.zeros_like(torch.from_numpy(pred_image).float()) - torch.from_numpy(pred_image2).float())))
            #loss = torch.sqrt(torch.mean((torch.from_numpy(pred_image).float() - torch.from_numpy(pred_image2).float())**2))  
            #loss /= torch.sqrt(torch.mean((torch.from_numpy(pred_image).float() - torch.from_numpy(pred_image2).float())**2)) 
            #loss /= torch.mean(torch.zeros_like(torch.from_numpy(pred_image).float()) - torch.from_numpy(pred_image2).float())
            #loss /= torch.sqrt(criterion(torch.zeros_like(torch.from_numpy(pred_image).float()), torch.from_numpy(pred_image2).float()))
            #loss /= torch.sqrt(criterion(255*torch.ones_like(torch.from_numpy(pred_image).float()), torch.from_numpy(pred_image2).float()))
            print(loss)  

            #import pdb; pdb.set_trace()           
            
            # # from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM                            
            # # # X: (N,3,H,W) a batch of non-negative RGB images (0 ~ 255)                    
            # # # Y: (N,3,H,W)       

            # # ssim_val = ssim(torch.from_numpy(pred_image).float().unsqueeze(0), torch.from_numpy(pred_image2).float().unsqueeze(0), size_average=True)  
            # # #ssim_val = ssim(torch.from_numpy(pred_image2).float().unsqueeze(0), torch.from_numpy(pred_image).float().unsqueeze(0), data_range=255, size_average=False)
            # # print(ssim_val.squeeze())

            # # #ssim_val = ssim(torch.zeros_like(torch.from_numpy(pred_image).float()).unsqueeze(0), torch.from_numpy(pred_image2).float().unsqueeze(0), data_range=255, size_average=False)
            # # #print(ssim_val.squeeze()) 

            # # from CannyEdgePytorch.net_canny import Net     
            # # #import imageio  

            # # def canny(raw_img, use_cuda=False): 
            # #     img = torch.from_numpy(raw_img.transpose((2, 0, 1)))
            # #     batch = torch.stack([img]).float()

            # #     net = Net(threshold=3.0, use_cuda=use_cuda)
            # #     if use_cuda:
            # #         net.cuda()
            # #     net.eval()

            # #     data = Variable(batch)
            # #     if use_cuda:
            # #         data = Variable(batch).cuda()

            # #     blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = net(data)

            # #     fig = plt.figure()
            # #     #fig.add_subplot(1,2,1)
            # #     #plt.imshow((thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float)) 
            # #     # # grad_mag.data.cpu().numpy()[0,0]  
            # #     #plt.imshow(grad_mag.data.cpu().numpy()[0,0])
            # #     plt.imshow(early_threshold.data.cpu().numpy()[0, 0])
            # #     plt.axis('off')
            # #     plt.show()
            # #     #plt.savefig('./theoutputnew'+str(counterr11)+'.png', bbox_inches='tight')  
            # #     plt.savefig('/home/ndionelis/pytorch_unet/theoutoutputnew'+str(counterr11)+'.png', bbox_inches='tight')

            # #     return early_threshold.data.cpu().numpy()[0, 0]  

            # #     #print(np.shape(raw_img)) 
            # #     #print(np.shape((thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float)))

            # #     #imageio.imwrite('gradient_magnitude.png',grad_mag.data.cpu().numpy()[0,0]) 
            # #     #imageio.imwrite('thin_edges.png', thresholded.data.cpu().numpy()[0, 0])
            # #     #imageio.imwrite('final.png', (thresholded.data.cpu().numpy()[0, 0] > 0.0).astype(float))
            # #     #imageio.imwrite('thresholded.png', early_threshold.data.cpu().numpy()[0, 0]) 

            # # #criterion = nn.MSELoss() 
            # # #loss = torch.sqrt(criterion(a, b))    
            # # loss = torch.abs(torch.mean(a - b))
            # # #b = canny2(((255*torch.ones_like(torch.from_numpy(mainimage))).numpy() / 255.0), use_cuda=True) 
            # # #b = canny2(((torch.zeros_like(torch.from_numpy(mainimage))).numpy() / 255.0), use_cuda=True)
            # # #b = np.resize(b, np.shape(a))
            # # #b = torch.from_numpy(b).float() 
            # # #loss /= torch.sqrt(criterion(a, b))    
            # # #loss /= torch.abs(torch.mean(a - b))
            # # print(loss) 

            # all_preds.append(pred) 
            # #all_gts.append(gt_e) 

            # #from scipy.misc import imread, imsave           
            # #from scipy.misc import imsave  
            # #import torch   
            # #from torch.autograd import Variable 
            # from CannyEdgePytorch.net_canny import Net
            # #import imageio 

            # #from CannyEdgePytorch.canny import *   
            # #from CannyEdgePytorch.net_canny import *   

            # import pytorch_ssim
            # #import torch
            # #from torch.autograd import Variable

            # #img1 = Variable(torch.rand(1, 1, 256, 256))   
            # #img2 = Variable(torch.rand(1, 1, 256, 256))

            # #if torch.cuda.is_available():   
            # #    img1 = img1.cuda()   
            # #    img2 = img2.cuda()  

            # criterion = nn.MSELoss() 
            # loss = torch.sqrt(criterion(a, b))
            # print(loss)

            foraverage += loss

            # from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
            # # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
            # # Y: (N,3,H,W)    

            # ssim_val = ssim(a, b, data_range=255, size_average=False)
            # print(ssim_val.squeeze())

            # foraverage2 += ssim_val.squeeze()

            # fig3 = plt.figure()
            # plt.imshow(convert_to_color(pred))
            # plt.axis('off')
            # #plt.show() 
            # #fig.add_subplot(1,3,3)       
            # #plt.imshow(gt)  
            # #plt.axis('off')   
            # #plt.show()  
            # #plt.savefig('./tesssttt2_tttlll2.png')                           
            # #io.imsave('./tesssttt2_tttlll2.png')             
            # #plt.pause(10)                    
            # #plt.figtext(0.5, 0.01, str(loss.item())+' '+str(ssim_val.squeeze().item())+' '+str(iou_pytorch(a, b).item()), wrap=True, horizontalalignment='center', fontsize=12)
            # #"{:.2f}".format(round(a, 2))             
            # plt.figtext(0.5, 0.01, "RMSE: "+str("{:.4f}".format(loss.item()))+', SSIM: '+str("{:.4f}".format(ssim_val.squeeze().item()))+', IoU: '+str("{:.4f}".format(iou_pytorch(a, b).item())), wrap=True, horizontalalignment='center', fontsize=12)
            # # try: 
            # #     plt.savefig('./oo/output'+str(counterr11)+'b.png', bbox_inches='tight')   
            # # except:  
            # #     os.mkdir('oo')    
            # #     plt.savefig('./oo/output'+str(counterr11)+'b.png', bbox_inches='tight')  
            # plt.savefig('./theoutput'+str(counterr11)+'b.png', bbox_inches='tight')

            plt.close('all')   

            torch.save(countertotal, 'countertotal.pt') 

            clear_output()
            torch.cuda.empty_cache()

            #time.sleep(4)  
            time.sleep(3)

            #clear_output()                          

            #metrics(pred.ravel(), gt_e.ravel())    
            #accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]), np.concatenate([p.ravel() for p in all_gts]).ravel())

            #break
            
        foraverage /= countertotal 

        print(foraverage)

    # print(foraverage) 
    # print(foraverage2)
    # print(foraverage3)

            #criterion = nn.MSELoss()
            #loss = torch.sqrt(criterion(a, b))
            #print(loss)

            #from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
            # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
            # Y: (N,3,H,W)   

            #ssim_val = ssim(a, b, data_range=255, size_average=False)
            #print(ssim_val.squeeze())

            fig3 = plt.figure()
            plt.imshow(convert_to_color(pred))
            plt.axis('off')
            #plt.show()
            #fig.add_subplot(1,3,3) 
            #plt.imshow(gt) 
            #plt.axis('off') 
            #plt.show() 
            #plt.savefig('./tesssttt2_tttlll2.png')   
            #io.imsave('./tesssttt2_tttlll2.png')    
            #plt.pause(10)          
            #plt.figtext(0.5, 0.01, str(loss.item())+' '+str(ssim_val.squeeze().item())+' '+str(iou_pytorch(a, b).item()), wrap=True, horizontalalignment='center', fontsize=12)
            #"{:.2f}".format(round(a, 2)) 
            try: 
                plt.savefig('./outputtss/output'+str(counterr11)+'b.png', bbox_inches='tight') 
            except:
                os.mkdir('outputtss')  
                plt.savefig('./outputtss/output'+str(counterr11)+'b.png', bbox_inches='tight')
            
            clear_output()
            torch.cuda.empty_cache()
            
            #clear_output()    

            #metrics(pred.ravel(), gt_e.ravel())
            #accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]), np.concatenate([p.ravel() for p in all_gts]).ravel())

            #break      

    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy 

from IPython.display import clear_output

def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch = 5):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    criterion = nn.NLLLoss2d(weight=weights)
    
    iter_ = 0 
    
    #print(loss) 
            
#train(net, optimizer, 50, scheduler)  

#net.load_state_dict(torch.load('./segnet_finale3'))   

# SegFormer
#net.load_state_dict(torch.load('./segformermain30082023')) 

# # torch.save(model.state_dict(), '/Data/ndionelis/formodels/sgfrmr2ehwh')               
# # data_dir = '/Data/ndionelis/bingmap/maindatasetnewnew2' 
# # Potsdam      
# # data_dir = '/Data/ndionelis/mainpo2'
# # On the dataset Vaihingen        
# # data_dir = '/Data/ndionelis/mainva2'  

# # net.load_state_dict(torch.load('./sgfrmr2023'))   

#clear_output()

#_, all_preds, all_gts = test(net, test_ids, all=True, stride=32)

