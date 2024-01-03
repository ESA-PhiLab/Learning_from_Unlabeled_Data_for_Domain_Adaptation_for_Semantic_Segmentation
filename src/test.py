#!/usr/bin/env python 
# coding: utf-8 

import sys 
#get_ipython().system('{sys.executable} -m pip install pytorch_ssim')
#get_ipython().system('{sys.executable} -m pip install pytorch_msssim')  
#pip install pytorch_ssim                 
 
# #!{sys.executable} -m conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# #!{sys.executable} -m pip install timm   

# # torch.save(model.state_dict(), '/Data/ndionelis/segformermain30082023noSubsaa22')
# # data_dir = '/Data/ndionelis/bingmap/mainTheDatasetNoSubsa2'

# # # This is to test: torch.save(model.state_dict(), './segformermain30082023')
# # # Also: data_dir = '../../CVUSA/bingmap/mainTheDataset'

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
# #BATCH_SIZE = 128  
# #BATCH_SIZE = 30 
# BATCH_SIZE = 10

# LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # # Label names
# N_CLASSES = len(LABELS) # # Number of classes 
# #print(N_CLASSES)  

# WEIGHTS = torch.ones(N_CLASSES) # # # Weights for class balancing 
# CACHE = True # # Store the dataset in-memory 

# #DATASET = 'Vaihingen'
# DATASET = 'Potsdam'    
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
# # The PyTorch imports    
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
import time
import kornia

WINDOW_SIZE = (256, 256) # # The patch size      
WINDOW_SIZE = (512, 512) # # Patch size    
STRIDE = 32 # # # # # Stride for testing 
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
#FOLDER = "./ISPRS_dataset/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
#FOLDER = "../../" 
FOLDER = "/Data/ndionelis/"
#BATCH_SIZE = 10 # # Number of samples in a mini-batch 
#BATCH_SIZE = 64
#BATCH_SIZE = 128    
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
    # # For RGB data
    DATA_FOLDER = MAIN_FOLDER + '2_Ortho_RGB/top_potsdam_{}_RGB.tif'
    LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
    ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'    
elif DATASET == 'Vaihingen':
    MAIN_FOLDER = FOLDER + 'Vaihingen/'
    DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
    LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
    ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'

# ISPRS color palette 
# # Let's define the standard ISPRS color palette
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)
invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
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

img = io.imread('/Data/ndionelis/Vaihingen/top/top_mosaic_09cm_area11.tif')
fig = plt.figure()
fig.add_subplot(121)
plt.imshow(img)

gt = io.imread('/Data/ndionelis/Vaihingen/gts_for_participants/top_mosaic_09cm_area11.tif')
fig.add_subplot(122) 
plt.imshow(gt)
plt.show()
 
array_gt = convert_from_color(gt)
print("Ground truth in numerical format has shape ({},{}) : \n".format(*array_gt.shape[:2]), array_gt)

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
           labels = range(len(label_values)))    
    for0 = 0  
    for1 = 0
    for2 = 0
    for3 = 0
    for4 = 0
    for5 = 0
    for6 = 0
    print("Confusion matrix :")
    print(cm)
    # # Compute global accuracy              
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    
    # # # Compute F1 score        
    # F1Score = np.zeros(len(label_values)) 
    # for i in range(len(label_values)):
    #     try:
    #         F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
    #     except:
    #         # # Ignore exception if there is no element in class i for test set 
    #         pass
    # print("F1Score :")
    # for l_id, score in enumerate(F1Score):
    #     print("{}: {}".format(label_values[l_id], score))

    # # # Compute kappa coefficient
    # total = np.sum(cm)
    # pa = np.trace(cm) / float(total)
    # pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    # kappa = (pa - pe) / (1 - pe);
    # print("Kappa: " + str(kappa))
    # return accuracy 

    totalfor0 = []
    totalfor0.append(for0)
    totalfor0.append(for1)
    totalfor0.append(for2)
    totalfor0.append(for3)
    totalfor0.append(for4)
    totalfor0.append(for5)            
    totalfor0.append(for6)

    # # Compute F1 score   
    totalF1Score = 0
    
    #print(label_values)  
    #F1Score = np.zeros(len(label_values))          
    F1Score = np.zeros(len(label_values))  

    for i in range(len(label_values)): 
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # # Ignore exception if there is no element in class i for test set          
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        #print(l_id)    

        #print("{}: {}, {}, {}".format(label_values[l_id], l_id, totalfor0[l_id], score))                           
        print("{}: {}, {}".format(label_values[l_id], totalfor0[l_id], score))  
        #print("{}: {}".format(label_values[l_id], score))  
        #totalF1Score += ((for0) / (for0 + for1 + for2 + for3 + for4 + for5)) * score
        totalF1Score += ((totalfor0[l_id]) / (sum(totalfor0))) * score 
        #if l_id < 5: 
        #    totalF1Score += ((totalfor0[l_id]) / (sum(totalfor0))) * score    

    # totalfor0 = []         
    # totalfor0.append(for0)     
    # totalfor0.append(for1)
    # totalfor0.append(for2)
    # totalfor0.append(for3)   
    # totalfor0.append(for4) 
    # totalfor0.append(for5)
    # totalfor0.append(for6)

    # print(for0)
    # print(for1)
    # print(for2)
    # print(for3)
    # print(for4)
    # print(for5)
    # #print(for6)   

    print(totalF1Score)
    
    #accuracy = totalF1Score

    mini = 1
    nclass = 6
    maxi = nclass
    nbins = nclass
    predict = predictions + 1  
    target = gts + 1  

    predict = torch.from_numpy(np.array(predict))    
    target = torch.from_numpy(np.array(target))

    #img = img.to(torch.float32)                 
    
    predict = predict.float() * (target < 7).float()  
    #predict = predict * (target < 7)                                            
    intersection = predict * (predict == target).float()  
    #intersection = predict * (predict == target)  
    # # areas of intersection and union     
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    target = target.float()
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    #return area_inter.float(), area_union.float() 
    #iou_main = area_inter.float() / area_union.float()  
    #iou_main = area_inter / area_union  
    
    #iou_main = area_inter.float() / area_union.float()     
    iou_main = 1.0 * np.sum(area_inter.float().numpy(), axis=0) / np.sum(np.spacing(1)+area_union.float().numpy(), axis=0)
    
    # mini = 1
    # #nclass = 6
    # nclass = 5
    # maxi = nclass
    # nbins = nclass
    # predict = predictions + 1  
    # target = gts + 1  

    # predict = torch.from_numpy(np.array(predict))    
    # target = torch.from_numpy(np.array(target))

    # #img = img.to(torch.float32)      
    
    # #predict = predict.float() * (target < 7).float()     
    # predict = predict.float() * (target < 6).float()  
    # #predict = predict * (target < 7)    
    # intersection = predict * (predict == target).float()  
    # #intersection = predict * (predict == target)  
    # # # areas of intersection and union     
    # # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    # area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    # area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    # target = target.float()
    # area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    # area_union = area_pred + area_lab - area_inter
    # assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    # #return area_inter.float(), area_union.float()  
    # #iou_main = area_inter.float() / area_union.float()  
    # #iou_main = area_inter / area_union  
    
    # #iou_main = area_inter.float() / area_union.float()                
    # iou_main = 1.0 * np.sum(area_inter.float().numpy(), axis=0) / np.sum(np.spacing(1)+area_union.float().numpy(), axis=0)

    #print(iou_main)    
    
    #print('')               
    
    print(iou_main)
    
    # #imPred = imPred * (imLab >= 0) 
    # #numClass = 6
    # numClass = 6 + 1
    
    # #imPred = predictions
    # #imLab = gts

    # imPred = predictions[gts <= 5] + 1   
    # imLab = gts[gts <= 5] + 1

    # #imPred = predictions[gts <= 5]     
    # #imLab = gts[gts <= 5]

    # #imPred = imPred * (imLab <= 5)    

    # imPred = imPred * (imLab <= 6)

    # # # Compute area intersection:
    # intersection = imPred * (imPred == imLab)
    # (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # # Compute area union:
    # (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    # (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    # area_union = area_pred + area_lab - area_intersection
    # #return (area_intersection, area_union)  
    # IoU = 1.0 * np.sum(area_intersection, axis=0) / np.sum(np.spacing(1)+area_union, axis=0)
    
    # print(IoU)

    #accuracy = iou_main   

    # # # Compute kappa coefficient    
    # total = np.sum(cm)
    # pa = np.trace(cm) / float(total)
    # pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    # kappa = (pa - pe) / (1 - pe);
    # print("Kappa: " + str(kappa)) 
    
    return accuracy

# # Dataset class
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
        
        # Decoder block 5   
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
        
        # Decoder block 5   
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
    # # Ignore missing keys 
    pass

# # Then, we load the network on GPU.
#net.cuda()

# For SegFormer
# We modify: https://github.com/huggingface/transformers/blob/v4.33.0/src/transformers/models/segformer/modeling_segformer.py#L746 

# The model SegFormer          
# # This is to test: torch.save(model.state_dict(), './segformermain30082023')
# # Also: data_dir = '../../CVUSA/bingmap/mainTheDataset'

# # We have modified: https://github.com/huggingface/transformers/blob/v4.33.0/src/transformers/models/segformer/modeling_segformer.py#L746 
#from transformers import SegformerForSemanticSegmentation

from transformers import SegformerForSemanticSegmentation
#import transformers 

id2label = {0 : 'Impervious surfaces',
            1 : 'Buildings',
            2 : 'Low vegetation',
            3 : 'Trees',
            4 : 'Cars',
            5 : 'Clutter'}
label2id = {v: k for k, v in id2label.items()}

# net = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
#                                                          num_labels=6, 
#                                                          id2label=id2label, 
#                                                          label2id=label2id,
#                                                          alpha=0.0,
# )   
# net = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
#                                                          num_labels=6, 
#                                                          id2label=id2label, 
#                                                          label2id=label2id,
#                                                          alpha=0.0,
# )

# Use B5
net = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                         num_labels=6, 
                                                         id2label=id2label, 
                                                         label2id=label2id,
                                                         alpha=0.0,
)
# # We use b5 

#net = SegNet()   
#net.cuda()
net.cuda()

# ### Loading the data      
# # # Load the datasets   
if DATASET == 'Potsdam':
    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
    all_ids = ["".join(f.split('')[5:7]) for f in all_files]
elif DATASET == 'Vaihingen':
    #all_ids = 
    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
    all_ids = [f.split('area')[-1].split('.')[0] for f in all_files]
test_ids = list(set(all_ids) - set(train_ids))
print("Tiles for training : ", train_ids)
print("Tiles for testing : ", test_ids)
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

# The optimizer 
#base_lr = 0.01 
#base_lr = 0.04
#base_lr = 0.08  
base_lr = 0.01

params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        params += [{'params':[value],'lr': base_lr}]
    else:
        params += [{'params':[value],'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # # # # # /home/nikolaos/CVUSA/bingmap   

    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    
    # # # # # /home/nikolaos/CVUSA/bingmap 
    #print(test_images)    
    #print(test_images.size) 
    
    all_preds = []
    all_gts = []
    
    # # # Switch the network to inference mode      
    #net.eval()
    
    net.eval()
    for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
        #pred = np.zeros(img.shape[:2] + (N_CLASSES,))

        #         fig = plt.figure()
        #         plt.imshow(np.asarray(255 * img, dtype='uint8'))
        #         plt.axis('off')
        #         plt.show()  
        #         print(img) 
        
        #print(img.shape) 
        
        #from PIL import Image
        #img2 = Image.open('/home/nikolaos/CVUSA/bingmap/0000001.jpg')
        
        #print(DATA_FOLDER.format(id)) 
        
        #img2 = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
        
        #img2 = (1 / 255 * np.asarray(io.imread('/home/nikolaos/CVUSA/bingmap/0000001.jpg'), dtype='float32') for id in test_ids)
        
        #         fig = plt.figure()
        #         #plt.imshow(np.asarray(255 * img2, dtype='uint8'))  
        #         #plt.imshow(img2)  
        #         plt.imshow(np.asarray(255 * img2, dtype='uint8'))
        #         plt.axis('off')
        #         plt.show()

        #         print(img2) 
        
        #print(img2.shape)
        
        #img = (1 / 255 * np.asarray(io.imread('/home/nikolaos/CVUSA/bingmap/0000001.jpg'), dtype='float32'))  
        
        #img = (1 / 255 * np.asarray(io.imread('/home/nikolaos/CVUSA/bingmap/19/0000017.jpg'), dtype='float32'))    
        
        countertotal = 0
        foraverage = 0
        foraverage2 = 0 
        foraverage3 = 0 
        
        #data_dir = '/Data/ndionelis/bingmap/main_dataset/val/1' 

        data_dir = '/Data/ndionelis/bingmap/mainTheDatasetNoSubsa2/val/1' 

        # # This is to test: torch.save(model.state_dict(), './segformermain30082023')    
        # # Also: data_dir = '../../CVUSA/bingmap/mainTheDataset'
        
        for file in os.listdir(data_dir):
            img = (1 / 255 * np.asarray(io.imread(os.path.join(data_dir, file)), dtype='float32'))
            
            countertotal += 1
            counterr11 = file
            #print(counterr11)                     
            
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
                image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
                image_patches = np.asarray(image_patches)
                #image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
                with torch.no_grad():
                    #image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)      
                    image_patches = Variable(torch.from_numpy(image_patches).cuda())
                    
                    # # /home/nikolaos/CVUSA/bingmsap              
                    
                    #encoding = feature_extractor(image_patches, return_tensors="pt")    
                    #pixel_values = encoding.pixel_values.to(device)  
                    
                    pixel_values = image_patches.to(device) 

                    outputs = net(pixel_values=pixel_values) 

                    logits = outputs.logits.cpu()

                    upsampled_logits = nn.functional.interpolate(logits,
                                    size=image_patches.shape[3], # # (height, width)   
                                    mode='bilinear',
                                    align_corners=False)
                    
                    #seg = upsampled_logits.argmax(dim=1)[0]
                    
                    outs = upsampled_logits.detach().cpu().numpy()

                    #print(image_patches.shape)
                    #print(image_patches)   

                    # # /home/nikolaos/CVUSA/bingmsap             

                    # # # # # Do the inference               
                    #outs = net(image_patches)    
                    #outs, _ = net(image_patches, 0.01)
                    #outs = net.forward2(image_patches)
                    
                    #outs = net.forward2(image_patches)
                    
                    #outputs = model(pixel_values=pixel_values, labels=labels)

                    #outs = outs.data.cpu().numpy()

                    # # # Fill in the results array     
                    for out, (x, y, w, h) in zip(outs, coords):
                        out = out.transpose((1,2,0))
                        pred[x:x+w, y:y+h] += out
                    del(outs)

                    #break

            pred = np.argmax(pred, axis=-1)

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
            print(counterr11)
            print(countertotal)
            counterr11 = counterr11[:-4]
            #print(counterr11)   
            try:
                plt.savefig('./inputs/input'+str(counterr11)+'.png', bbox_inches='tight')
            except:
                os.mkdir('inputs')
                plt.savefig('./inputs/input'+str(counterr11)+'.png', bbox_inches='tight')
            #plt.savefig('inputim1.png', bbox_inches='tight')                    
            fig2 = plt.figure() 
            plt.imshow(convert_to_color(pred))
            plt.axis('off')
            #plt.show()
            #fig.add_subplot(1,3,3) 
            #plt.imshow(gt)
            #plt.axis('off')      
            #plt.show() 
            #plt.pause(10)    
            try:
                plt.savefig('./outputs/output'+str(counterr11)+'.png', bbox_inches='tight')
            except:
                os.mkdir('outputs')
                plt.savefig('./outputs/output'+str(counterr11)+'.png', bbox_inches='tight')
            #plt.savefig('outputim1.png', bbox_inches='tight')      

            all_preds.append(pred)
            #all_gts.append(gt_e) 

            import pytorch_ssim
            #import torch
            #from torch.autograd import Variable

            #img1 = Variable(torch.rand(1, 1, 256, 256))
            #img2 = Variable(torch.rand(1, 1, 256, 256))

            from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
            # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
            # Y: (N,3,H,W)    
            
            fig3 = plt.figure()
            plt.imshow(convert_to_color(pred))
            plt.axis('off')
            #plt.show()
            #fig.add_subplot(1,3,3)   
            #plt.imshow(gt)  
            #plt.axis('off')  
            #plt.show()  
            #plt.pause(10)                    
            #plt.figtext(0.5, 0.01, str(loss.item())+' '+str(ssim_val.squeeze().item())+' '+str(iou_pytorch(a, b).item()), wrap=True, horizontalalignment='center', fontsize=12)
            #"{:.2f}".format(round(a, 2))            
            plt.figtext(0.5, 0.01, "RMSE: "+str("{:.4f}".format(loss.item()))+', SSIM: '+str("{:.4f}".format(ssim_val.squeeze().item()))+', IoU: '+str("{:.4f}".format(iou_pytorch(a, b).item())), wrap=True, horizontalalignment='center', fontsize=12)
            try: 
                plt.savefig('./outputs/output'+str(counterr11)+'b.png', bbox_inches='tight') 
            except:  
                os.mkdir('outputs')   
                plt.savefig('./outputs/output'+str(counterr11)+'b.png', bbox_inches='tight')
                
            plt.close('all')

            clear_output()
            torch.cuda.empty_cache()
            
            time.sleep(5)

            #clear_output()                     

            #metrics(pred.ravel(), gt_e.ravel())    
            #accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]), np.concatenate([p.ravel() for p in all_gts]).ravel())

            #break
            
    foraverage /= countertotal
    foraverage2 /= countertotal 
    foraverage3 /= countertotal 
    
    print(foraverage)
    print(foraverage2)
    print(foraverage3)

    #clear_output()    

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
    
#train(net, optimizer, 10, scheduler) 

#net.load_state_dict(torch.load('./segformermain30082023'))
#net.load_state_dict(torch.load('./segnet_finale'))   

#net.load_state_dict(torch.load('./segformermain300820232'))      

net.load_state_dict(torch.load('/Data/ndionelis/formodels/srmr2ehw4h44'))

#DATASET = 'Vaihingen'  
DATASET = 'Potsdam'  

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

print("Tiles for testing : ", test_ids)

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []
    
    net.eval()
    for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (N_CLASSES,))

        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False)):
            if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                    _pred = np.argmax(pred, axis=-1)
                    fig = plt.figure()
                    fig.add_subplot(1,3,1)
                    plt.imshow(np.asarray(255 * img, dtype='uint8'))
                    fig.add_subplot(1,3,2)
                    plt.imshow(convert_to_color(_pred))
                    fig.add_subplot(1,3,3)
                    plt.imshow(gt)
                    clear_output()
                    plt.show()
                    
            image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
            image_patches = np.asarray(image_patches)
            #image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
            
            with torch.no_grad():
                image_patches = Variable(torch.from_numpy(image_patches).cuda())

                # # Do the inference   
                # outs = net(image_patches)
                # outs = outs.data.cpu().numpy()

                pixel_values = image_patches.to(device)
                outputs = net(pixel_values=pixel_values) 

                logits = outputs.logits.cpu()
                upsampled_logits = nn.functional.interpolate(logits,
                                size=image_patches.shape[3], # # (height, width)   
                                mode='bilinear',
                                align_corners=False)
                
                #print(upsampled_logits.shape) 
                # # torch.Size([10, 150, 256, 256])       
                
                #seg = upsampled_logits.argmax(dim=1)[0] 
                
                outs = upsampled_logits.detach().cpu().numpy()
                
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1,2,0))
                    pred[x:x+w, y:y+h] += out
                del(outs)  

        pred = np.argmax(pred, axis=-1)

        clear_output() 
        
        fig = plt.figure()
        #fig.add_subplot(1,3,1)  
        plt.imshow(np.asarray(255 * img, dtype='uint8'))
        plt.axis('off')
        plt.savefig('Input1.png', bbox_inches='tight')
        
        fig = plt.figure() 
        #fig.add_subplot(1,3,2)    
        plt.imshow(convert_to_color(pred))
        plt.axis('off')
        plt.savefig('Output1.png', bbox_inches='tight')

        fig = plt.figure() 
        plt.imshow(gt)
        plt.axis('off')
        plt.savefig('Correct1.png', bbox_inches='tight')

        all_preds.append(pred) 
        all_gts.append(gt_e)

        clear_output()
        
        metrics(pred.ravel(), gt_e.ravel())
        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]), np.concatenate([p.ravel() for p in all_gts]).ravel())

    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy

#_, all_preds, all_gts = test(net, test_ids, all=True, stride=32)      
 
accAccuracy, all_preds, all_gts = test(net, test_ids, all=True, stride=32)    

print('') 
#print(accAccuracy)  

print(accAccuracy)

for p, id_ in zip(all_gts, test_ids):
    img = convert_to_color(p)
    plt.imshow(img) and plt.show()
    #io.imsave('./inference_tile_{}.png'.format(id_), img)      
    #io.imsave('./testing2_tile2_{}.png'.format(id_), img) 
    #io.imsave('./tst1_tl1_{}.png'.format(id_), img)
    #io.imsave('./test2_tl2_{}.png'.format(id_), img) 
    io.imsave('./testt2_ttll2_{}.png'.format(id_), img)

#_, all_preds, all_gts = test(net, test_ids, all=True, stride=32)

