#!/usr/bin/env python
# coding: utf-8

# Define PyTorch dataset and dataloaders
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import numpy as np
import random

# We set the random seed  
#SEED = 17  
#SEED = 71 
SEED = random.randint(1, 10000)
print(SEED)
torch.cuda.empty_cache()
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

#NUMWORKERS = 6  
NUMWORKERS = 0                                  

# We have modified: https://github.com/huggingface/transformers/blob/v4.33.0/src/transformers/models/segformer/modeling_segformer.py#L746 

import requests, zipfile, io  
def download_data(): 
    url = "https://www.dropbox.com/s/l1e45oht447053f/ADE20k_toy_dataset.zip?dl=1"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
download_data()
from datasets import load_dataset
load_entire_dataset = False
if load_entire_dataset:
  dataset = load_dataset("scene_parse_150")

class SemanticSegmentationDataset(Dataset):
    def __init__(self, root_dir, feature_extractor, train=True):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train = train
        sub_path = "training" if self.train else "validation"
        self.img_dir = os.path.join(self.root_dir, "images", sub_path)
        self.ann_dir = os.path.join(self.root_dir, "annotations", sub_path)
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
          image_file_names.extend(files)
        self.images = sorted(image_file_names)
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
          annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)
        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")
        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_()
        return encoded_inputs

from transformers import SegformerFeatureExtractor
#import transformers
root_dir = './ADE20k_toy_dataset'
feature_extractor = SegformerFeatureExtractor(reduce_labels=True)
train_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor)
valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, feature_extractor=feature_extractor, train=False)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(valid_dataset))

encoded_inputs = train_dataset[0]
encoded_inputs["pixel_values"].shape
encoded_inputs["labels"].shape
encoded_inputs["labels"]
encoded_inputs["labels"].squeeze().unique()

# Define corresponding dataloaders
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=2)
batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k, v.shape)
batch["labels"].shape

mask = (batch["labels"] != 255)
mask

batch["labels"][mask]

# We define the model
from transformers import SegformerForSemanticSegmentation
import json
from huggingface_hub import cached_download, hf_hub_url

# # # load id2label mapping from a JSON on the hub  
# repo_id = "datasets/huggingface/label-files"
# filename = "ade20k-id2label.json"

# id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
# id2label = {int(k): v for k, v in id2label.items()}
# label2id = {v: k for k, v in id2label.items()}

id2label = json.load(open("./ade20k-id2label.json", "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

# Define our model   
# model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
#                                                          num_labels=6, 
#                                                          id2label=id2label, 
#                                                          label2id=label2id,
# )

from datasets import load_metric
metric = load_metric("mean_iou")

#import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
from skimage import io
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
#import random
import itertools
# # Matplotlib           
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')                 
#from IPython import get_ipython    
#get_ipython().run_line_magic('matplotlib', 'inline')  
#exec(%matplotlib inline)
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

# # Parameters   
WINDOW_SIZE = (256, 256) # # Patch size   
STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # # Number of input channels (e.g. RGB) 
#FOLDER = "./ISPRS_dataset/" # Replace with your "/path/to/the/ISPRS/dataset/folder/"
#FOLDER = "../../"
FOLDER = "/Data/ndionelis/"
#BATCH_SIZE = 64  
#BATCH_SIZE = 128  
BATCH_SIZE = 10
#BATCH_SIZE = 30  

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # # Label names
N_CLASSES = len(LABELS) # # Number of classes  
#print(N_CLASSES)   

WEIGHTS = torch.ones(N_CLASSES) # # # Weights for class balancing  
CACHE = True # # # Store the dataset in-memory  

#DATASET = 'Vaihingen' 
DATASET = 'Potsdam'    

MAIN_FOLDER = FOLDER + 'Potsdam/'
# # Uncomment the next line for IRRG data     
# DATA_FOLDER = MAIN_FOLDER + '3_Ortho_IRRG/top_potsdam_{}_IRRG.tif'     
# # For RGB data    
#print(MAIN_FOLDER)
DATA_FOLDER = MAIN_FOLDER + '2_Ortho_RGB/top_potsdam_{}_RGB.tif'
#LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
LABEL_FOLDER = MAIN_FOLDER + 'top_potsdam_{}_label.tif'
#ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'
ERODED_FOLDER = MAIN_FOLDER + 'top_potsdam_{}_label_noBoundary.tif'    

# if DATASET == 'Potsdam':
#     MAIN_FOLDER = FOLDER + 'Potsdam/'
#     # # Uncomment the next line for IRRG data     
#     # DATA_FOLDER = MAIN_FOLDER + '3_Ortho_IRRG/top_potsdam_{}_IRRG.tif'     
#     # # For RGB data    
#     #print(MAIN_FOLDER)
#     #sadfszf
#     DATA_FOLDER = MAIN_FOLDER + '2_Ortho_RGB/top_potsdam_{}_RGB.tif'
#     #LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants/top_potsdam_{}_label.tif'
#     LABEL_FOLDER = MAIN_FOLDER + 'top_potsdam_{}_label.tif'
#     #ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif'
#     ERODED_FOLDER = MAIN_FOLDER + 'top_potsdam_{}_label_noBoundary.tif'    

# elif DATASET == 'Vaihingen':
#     MAIN_FOLDER = FOLDER + 'Vaihingen/'
#     #print(MAIN_FOLDER) 
#     #asdfszdf
#     DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
#     LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
#     ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'

# # ISPRS color palette 
# # # Let's define the standard ISPRS color palette 
palette = {0 : (255, 255, 255), # Impervious surfaces (white) 
           1 : (0, 0, 255),     # # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black) 
invert_palette = {v: k for k, v in palette.items()}

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
    # # Compute global accuracy  
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    # # # Compute F1 score  
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # # Ignore exception if there is no element in class i for test set 
            #pass  
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))
    # # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe);
    print("Kappa: " + str(kappa))
    return accuracy 

# Load the dataset 
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
        
        #data_p, label_p = self.data_augmentation(data_p, label_p)  
        data_p, label_p = self.data_augmentation(data_p, label_p)

        # # # # # Return the torch.Tensor values  
        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))

# model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
#                                                          num_labels=6, 
#                                                          id2label=id2label, 
#                                                          label2id=label2id,
# )     
    
id2label = {0 : 'Impervious surfaces',
            1 : 'Buildings',
            2 : 'Low vegetation',
            3 : 'Trees',
            4 : 'Cars',
            5 : 'Clutter'}
label2id = {v: k for k, v in id2label.items()}

# model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
#                                                          num_labels=6, 
#                                                          id2label=id2label, 
#                                                          label2id=label2id,
# )

# We use B5  
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                         num_labels=6, 
                                                         id2label=id2label, 
                                                         label2id=label2id,
                                                         alpha=0.0,
)
# Use b5 

# Load the data   
if DATASET == 'Potsdam':
    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
    #print(all_files) 
    #all_ids = ["".join(f.split('')[5:7]) for f in all_files] 
    #print(all_ids)    

elif DATASET == 'Vaihingen':
    #all_ids = 
    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
    all_ids = [f.split('area')[-1].split('.')[0] for f in all_files]
    
    #print(all_files) 
    #print(all_ids)

#folderName = '../../MainPotsd'
folderName = '/Data/ndionelis/MainPotsd'
names_data = os.listdir(folderName) # to not load all data in a single tensor, load only the names 
length_names = len(names_data)

from torchvision import transforms, datasets
import shutil 
     
data_dir = '/Data/ndionelis/mainpo4244'

train_ids = []
iTheLoopNumber = 0
for iTheLoop in training_data:
    train_ids.append(training_data[iTheLoopNumber][12:-8])
    iTheLoopNumber += 1

print("Tiles for training : ", train_ids)     
train_set = ISPRS_dataset(train_ids, cache=CACHE)  
#print(len(train_set))                                      
#train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)     
#train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE) 
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUMWORKERS)

# train_dataloader_iter = iter(train_dataloader)         
# #train_dataloader_iter_next = next(train_dataloader_iter)            
# #train_dataloader_iter_next1, train_dataloader_iter_next2 = train_dataloader_iter_next      
# train_dataloader_iter_next1, train_dataloader_iter_next2 = next(train_dataloader_iter)
# print(train_dataloader_iter_next1)
# print(train_dataloader_iter_next2)
# print(train_dataloader_iter_next1.shape)
# print(train_dataloader_iter_next2.shape)

# train_dataloader_iter = iter(train_dataloader)    
# #train_dataloader_iter_next = next(train_dataloader_iter)    
# #train_dataloader_iter_next1, train_dataloader_iter_next2 = train_dataloader_iter_next   
# train_dataloader_iter_next1, train_dataloader_iter_next2 = next(train_dataloader_iter)
# print(train_dataloader_iter_next1)
# print(train_dataloader_iter_next2)
# print(train_dataloader_iter_next1.shape)
# print(train_dataloader_iter_next2.shape)
# # torch.Size([10, 3, 256, 256])    
# # torch.Size([10, 256, 256])

# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomCrop((256, 256)),
#         transforms.ToTensor(),
#     ]),
#     'val': transforms.Compose([
#         transforms.RandomCrop((256, 256)),
#         transforms.ToTensor(),
#     ]),
# }  
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                         data_transforms[x])
#                 for x in ['train', 'val']}

# #print(image_datasets['train'].imgs) 
# #print(image_datasets['train'].targets)

# #print(dataloaders['train'])     
# dataloaders_iter = iter(dataloaders['train'])
# dataloaders_iter_next1, dataloaders_iter_next2 = next(dataloaders_iter)
# print(dataloaders_iter_next1)
# print(dataloaders_iter_next2)
# print(dataloaders_iter_next1.shape)
# print(dataloaders_iter_next2.shape)

DATASET = 'Vaihingen'    
#DATASET = 'Potsdam'      

MAIN_FOLDER = FOLDER + 'Vaihingen/'
DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
#LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
LABEL_FOLDER = MAIN_FOLDER + 'top_mosaic_09cm_area{}.tif'
#ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'
ERODED_FOLDER = MAIN_FOLDER + 'top_mosaic_09cm_area{}_noBoundary.tif'

# if DATASET == 'Potsdam':   
#     MAIN_FOLDER = FOLDER + 'Potsdam/'
#     # Uncomment the next line for IRRG data     
#     # DATA_FOLDER = MAIN_FOLDER + '3_Ortho_IRRG/top_potsdam_{}_IRRG.tif'     
#     # # For RGB data    
#     #print(MAIN_FOLDER)
#     #sadfszf
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
        # # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)
        
        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = 1/255 * np.asarray(io.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data
            
        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else: 
            # # Labels are converted from RGB to their numeric values     
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2,y1:y2]
        label_p = label[x1:x2,y1:y2]
        
        data_p, label_p = self.data_augmentation(data_p, label_p)         

        return (torch.from_numpy(data_p),
                torch.from_numpy(label_p))

# Load the datasets   
if DATASET == 'Potsdam':
    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
    #print(all_files)

    #all_ids = ["".join(f.split('')[5:7]) for f in all_files]
    #print(all_ids)  

elif DATASET == 'Vaihingen':
    #all_ids = 
    all_files = sorted(glob(LABEL_FOLDER.replace('{}', '*')))
    all_ids = [f.split('area')[-1].split('.')[0] for f in all_files]
    
    #print(all_files) 
    #print(all_ids)

# train_loader2_iter = iter(train_loader2)
# train_loader2_iter_next1, train_loader2_iter_next2 = next(train_loader2_iter)
# print(train_loader2_iter_next1)
# print(train_loader2_iter_next2)
# print(train_loader2_iter_next1.shape)
# print(train_loader2_iter_next2.shape)
# # torch.Size([10, 3, 256, 256])
# # torch.Size([10, 256, 256])

#folderName = '../../Vaihingen/top'     
folderName = '/Data/ndionelis/Vaihingen/top'
names_data = os.listdir(folderName) # to not load all data in a single tensor, load only the names 
length_names = len(names_data)

from torchvision import transforms, datasets
#import shutil 
       
#data_dir = '/Data/ndionelis/mainva2' 
data_dir = '/Data/ndionelis/mainva4244'

train_ids = []
iTheLoopNumber = 0
for iTheLoop in training_data:
    train_ids.append(training_data[iTheLoopNumber][20:-4])    
    iTheLoopNumber += 1
#print(train_ids)     

print("Tiles for training : ", train_ids) 
#print("Tiles for testing : ", test_ids)

train_set = ISPRS_dataset(train_ids, cache=CACHE)  
#print(len(train_set))                                     
#train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)  
#train_loader2 = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)
train_loader2 = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUMWORKERS)

# train_loader2_iter = iter(train_loader2)
# train_loader2_iter_next1, train_loader2_iter_next2 = next(train_loader2_iter)
# print(train_loader2_iter_next1)
# print(train_loader2_iter_next2)
# print(train_loader2_iter_next1.shape)
# print(train_loader2_iter_next2.shape)
# # torch.Size([10, 3, 256, 256])
# # torch.Size([10, 256, 256])

# data_transforms = {
#     'train': transforms.Compose([
#         transforms.ToTensor(),
#     ]),
#     'val': transforms.Compose([
#         transforms.ToTensor(),
#     ]),
# }  
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                         data_transforms[x])
#                 for x in ['train', 'val']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
#                                             shuffle=True, num_workers=8) 
#             for x in ['train', 'val']} 
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes

# Define the optimizer                
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)           
#optimizer = torch.optim.AdamW(model.parameters(), lr=0.0000006)  

optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)

folderName = '/Data/ndionelis/bingmap/19'  
#folderName = '../../CVUSA/bingmap/19'

names_data = os.listdir(folderName) # # to not load all data in a single tensor, load only the names             
length_names = len(names_data)
#print(names_data)
#print(length_names)  

# print(training_data)
# print(len(training_data))
# print(test_data)
# print(len(test_data))

from torchvision import transforms, datasets

#data_transforms = transforms.Compose([transforms.ToTensor(),])        
#data_transforms = transforms.ToTensor()   

#import shutil   
    
#data_dir = '../../CVUSA/bingmap/mainTheDatasetNoSubsaa223newnew' 
#data_dir = '/Data/ndionelis/bingmap/maindatasetnewnew2'       
data_dir = '/Data/ndionelis/bingmap/maindatasetnewnew4244'  

#data_dir = '/Data/ndionelis/bingmap/mainDataset' 
# image_datasets = {x: datasets.ImageFolder(data_dir,
#                                       data_transforms)
#               if x in training_data;
#               else: continue}   
#image_datasets = datasets.ImageFolder(os.path.join(data_dir, training_data), data_transforms)
#image_datasets2 = datasets.ImageFolder(os.path.join(data_dir, test_data), data_transforms)

data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
}  

#data_dir = 'data/hymenoptera_data'     
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                for x in ['train', 'val']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
#                                             shuffle=True, num_workers=8) 
#             for x in ['train', 'val']}
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
#                                             shuffle=True, num_workers=6) 
#             for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=NUMWORKERS) 
            for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move model to GPU                     
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)     

model.to(device)
model.train()     
   
#model.load_state_dict(torch.load('/Data/ndionelis/segformermain30082023')) 

w, h = WINDOW_SIZE
 
for epoch in range(200): 
   print("Epoch:", epoch)
   #train_loader2_iter = iter(train_loader2)    
   train_loader2_iter = iter(train_loader2)
   dataloaders_iter = iter(dataloaders['train'])
   #for idx, batch in enumerate(tqdm(train_dataloader)):      
   for idx, (pixel_values, labels) in enumerate(tqdm(train_dataloader)):
        pixel_values, labels = Variable(pixel_values.cuda()), Variable(labels.cuda())
        epochs = 200
        #epochs = 50   
        data_dataloaders = next(dataloaders_iter)
        #data_dataloaders2 = next(dataloaders_iter2)   
        #data_dataloaders, labels_dataloaders = data_dataloaders    
        data_dataloaders, _ = data_dataloaders 
        #w, h = WINDOW_SIZE
        W, H = data_dataloaders.shape[-2:]
        # x1 = random.randint(0, W - w - 1) 
        # x2 = x1 + w
        # y1 = random.randint(0, H - h - 1) 
        # y2 = y1 + h
        # data_dataloaders = data_dataloaders[:, :, x1:x2, y1:y2] 
        for iToLoop in range(BATCH_SIZE):
            if iToLoop == 0:
                image_patches = torch.zeros((BATCH_SIZE, 3, w, h), device=device)
            x1 = random.randint(0, W - w - 1)
            x2 = x1 + w
            y1 = random.randint(0, H - h - 1)
            y2 = y1 + h 
            #data_dataloaders = data_dataloaders[:, :, x1:x2, y1:y2]  
            #data_dataloaders = data_dataloaders[iToLoop, :, x1:x2, y1:y2]
            image_patches[iToLoop, :, :, :] = data_dataloaders[iToLoop, :, x1:x2, y1:y2]
        
        #p = float(idx + e * len(train_dataloader)) / epochs / len(train_dataloader)    
        p = float(idx + (epoch+1) * len(train_dataloader)) / epochs / len(train_dataloader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        #encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")                             
        #pixel_values = encoded_inputs.pixel_values.to(device)
        #labels = encoded_inputs.labels.to(device)

        #image_patches = Variable(torch.from_numpy(image_patches).cuda())       
        #encoding = feature_extractor(image_patches, return_tensors="pt")           
        #pixel_values = encoding.pixel_values.to(device)

        # # # # # get the inputs        
        #pixel_values = batch["pixel_values"].to(device)           
        #labels = batch["labels"].to(device)
        #print(pixel_values.shape) 
        #print(labels.shape) 
        
        #         #import matplotlib.pyplot as plt      
        #         plt.figure()       
        #         plt.imshow(pixel_values[0,:,:,:].permute(1, 2, 0).cpu())    
        #         plt.show()  

        #         plt.figure()
        #         plt.imshow((labels.unsqueeze(1))[0,:,:,:].permute(1, 2, 0).cpu()) 
        #         plt.show()

        #         plt.figure()
        #         plt.imshow(pixel_values[1,:,:,:].permute(1, 2, 0).cpu())
        #         plt.show()

        #         plt.figure()
        #         plt.imshow((labels.unsqueeze(1))[1,:,:,:].permute(1, 2, 0).cpu())
        #         plt.show()
        #         # # torch.Size([10, 3, 256, 256])  
        #         # # torch.Size([10, 256, 256])
        
        # # torch.Size([2, 3, 512, 512])     
        # # torch.Size([2, 512, 512]) 
        
        # # # zero the parameter gradients         
        optimizer.zero_grad()

        data2, target2 = next(train_loader2_iter) 
        data2, target2 = Variable(data2.cuda()), Variable(target2.cuda())
        #print(alpha)          

        # import matplotlib.pyplot as plt
        # plt.imshow(pixel_values[3,:,:,:].permute(1, 2, 0).cpu().numpy())
        # plt.savefig('beforeIm.png') 

        # # radias = int(0.1 * 256) // 2
        # # kernel_size = radias * 2 + 1
        # # blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
        # #                         stride=1, padding=0, bias=False, groups=3)
        # # blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
        # #                         stride=1, padding=0, bias=False, groups=3)
        # # #k = kernel_size 
        # # #r = radias

        # # blur = nn.Sequential(
        # #     nn.ReflectionPad2d(radias),
        # #     blur_h,
        # #     blur_v
        # # )

        # # #self.pil_to_tensor = transforms.ToTensor()  
        # # #self.tensor_to_pil = transforms.ToPILImage()   

        # # #img = self.pil_to_tensor(img).unsqueeze(0)          

        # # sigma = np.random.uniform(0.1, 2.0)
        # # x = np.arange(-radias, radias + 1)
        # # x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        # # x = x / x.sum()
        # # x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        # # blur_h.weight.data.copy_(x.view(3, 1, kernel_size, 1))   
        # # blur_v.weight.data.copy_(x.view(3, 1, 1, kernel_size))
        # # pixel_values = blur(pixel_values) 

        # blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        # pixel_values = blurrer(pixel_values)

        # plt.figure()
        # plt.imshow(pixel_values[3,:,:,:].permute(1, 2, 0).cpu().numpy())
        # plt.savefig('afterImage.png')
        
        # if epoch > 8:   
        #     if random.random() < 0.5:   
        #         # # # # Gaussian blur                                                               
        #         #applier = T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5)
        #         #applier = T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=1.0)
        #         applier = T.RandomVerticalFlip(p=1.0) 
        #         pixel_values = applier(pixel_values) 
        #         labels = applier(labels)
        #         image_patches = applier(image_patches)
        #         data2 = applier(data2)
        #         target2 = applier(target2) 
            
        #     if random.random() < 0.5: 
        #         # # # Gaussian blur                                                                         
        #         #applier = T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5)
        #         #applier = T.RandomApply(transforms=[T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=1.0)
        #         applier = T.RandomHorizontalFlip(p=1.0)
        #         pixel_values = applier(pixel_values)  
        #         labels = applier(labels) 
        #         image_patches = applier(image_patches)
        #         data2 = applier(data2) 
        #         target2 = applier(target2)
        
        #     if random.random() < 0.5:
        #         blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        #         pixel_values = blurrer(pixel_values)
        #         image_patches = blurrer(image_patches)
        #         data2 = blurrer(data2)

        #print(pixel_values)
        #print(pixel_values.shape) 

        # We modify: https://github.com/huggingface/transformers/blob/v4.33.0/src/transformers/models/segformer/modeling_segformer.py#L746 

        # # forward + backward + optimize                                                    
        #outputs = model(pixel_values=pixel_values, labels=labels, alpha=alpha)          
        outputs = model(pixel_values=pixel_values, labels=labels, alpha=alpha, image_patches=image_patches, data2=data2, target2=target2)
        #outputs = model(pixel_values=pixel_values, labels=labels)            

        # For the above, we have modified: https://github.com/huggingface/transformers/blob/v4.33.0/src/transformers/models/segformer/modeling_segformer.py#L746 

        loss, logits = outputs.loss, outputs.logits 
        
        # output2, oo222 = net(data2, alpha)   
        # #loss += 10.0 * CrossEntropy2d(output2, target2, weight=weights)        
        # loss += CrossEntropy2d(output2, target2, weight=weights)
        # loss += smp.losses.JaccardLoss(mode="multiclass", classes=6)(y_pred=output2, y_true=target2) 
        # domain_label2 = (2*torch.ones(BATCH_SIZE).long()).cuda()
        # #err_s_domain = nn.CrossEntropyLoss()(oo222, domain_label2)                            
        # loss += nn.CrossEntropyLoss()(oo222, domain_label2)
        #loss.backward()                
        
        loss.backward()
        optimizer.step()
        print("Loss:", loss.item()) 
        
        # # # # # evaluate       
        # with torch.no_grad():  
        #   upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        #   predicted = upsampled_logits.argmax(dim=1) 
          
        #   # # note that the metric expects predictions + labels as numpy arrays 
        #   metric.add_batch(predictions = predicted.detach().cpu().numpy(), references = labels.detach().cpu().numpy())

        # # # # let's print loss and metrics every 100 batches
        # #if idx % 100 == 0:
        # if idx % 10 == 0:
        #   metrics = metric._compute(predictions = predicted.detach().cpu().numpy(), references = labels.detach().cpu().numpy(), num_labels=len(id2label), 
        #                            ignore_index=255,
        #                            reduce_labels=False, # we've already reduced the labels before)
        #   )

        #   print("Loss:", loss.item())
        #   print("Mean_iou:", metrics["mean_iou"])
        #   print("Mean accuracy:", metrics["mean_accuracy"])


# ## Inference         
##image = Image.open('./ADE20k_toy_dataset/images/training/ADE_train_00000001.jpg')
 
def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
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

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # # /home/nikolaos/CVUSA/bingmap          

    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    
    # # /home/nikolaos/CVUSA/bingmap          
    #print(test_images)   
    #print(test_images.size)   

    all_preds = []
    all_gts = []
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
        
        # #img = (1 / 255 * np.asarray(io.imread('/Data/ndionelis/bingmap/19/'+str(44506).zfill(7)+'.jpg'), dtype='float32'))
        
        # #img = (1 / 255 * np.asarray(io.imread('/home/nikolaos/CVUSA/bingmap/19/0000019.jpg'), dtype='float32'))                                            
        
        # #img = (1 / 255 * np.asarray(io.imread('/home/nikolaos/CVUSA/bingmap/19/0000020.jpg'), dtype='float32'))

        # #image  
        
        # #plt.figure
        # #plt.imshow(img)
        # #plt.show()
        
        pred = np.zeros(img.shape[:2] + (N_CLASSES,))
        
        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False)):
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
                    
            image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
            image_patches = np.asarray(image_patches)
            with torch.no_grad():
                image_patches = Variable(torch.from_numpy(image_patches).cuda())
                
                #print(image_patches.shape)    
                #print(image_patches)   
                
                # torch.Size([10, 3, 256, 256])     
                # tensor([[[[0.0980, 0.0902, 0.0784,  ..., 0.1176, 0.1137, 0.1020],
                # [0.0863, 0.0706, 0.0549,  ..., 0.1098, 0.1059, 0.1020],
                
                # # /home/nikolaos/CVUSA/bingmsap              
                
                #             from PIL import Image  
                #             img2 = Image.open('/home/nikolaos/CVUSA/bingmap/0000001.jpg')

                #             from torchvision import transforms 
                #             img2 = transforms.Resize((256, 256))(img2)

                #             #image_patches = transforms.ToTensor()(img)
                #             image_patches = transforms.ToTensor()(img2).unsqueeze_(0)

                #             for iloop in range(2, 10):
                #                 img2 = Image.open('/home/nikolaos/CVUSA/bingmap/000000'+str(iloop)+'.jpg')
                #                 img2 = transforms.Resize((256, 256))(img2)
                #                 img2 = transforms.ToTensor()(img2).unsqueeze_(0)
                #                 image_patches = torch.cat((image_patches, img2), 0)

                #             img2 = Image.open('/home/nikolaos/CVUSA/bingmap/0000010.jpg')
                #             img2 = transforms.Resize((256, 256))(img2)
                #             img2 = transforms.ToTensor()(img2).unsqueeze_(0)
                #             image_patches = torch.cat((image_patches, img2), 0).cuda()

                #             #print(image_patches.shape)
                #             #print(image_patches)        

                # # Do the inference          
                #outs = net(image_patches)   
                #outs, _ = net(image_patches, 0.01) 
                #outs = net.forward2(image_patches)
                #outs = outs.data.cpu().numpy() 
                
                # # prepare the image for the model 
                #encoding = feature_extractor(image_patches, return_tensors="pt")
                #pixel_values = encoding.pixel_values.to(device)
                #print(pixel_values.shape)

                #encoding = feature_extractor(image_patches, return_tensors="pt")
                #pixel_values = encoding.pixel_values.to(device)
                
                pixel_values = image_patches.to(device) 
                outputs = model(pixel_values=pixel_values) 

                logits = outputs.logits.cpu()

                upsampled_logits = nn.functional.interpolate(logits,
                                size=image_patches.shape[3], # # (height, width)
                                mode='bilinear',
                                align_corners=False)
                
                #seg = upsampled_logits.argmax(dim=1)[0] 
                outs = upsampled_logits.detach().cpu().numpy()
                
                #             color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # # height, width, 3
                #             palette = np.array(ade_palette())
                #             for label, color in enumerate(palette):
                #                 color_seg[seg == label, :] = color
                #             # # Convert to BGR
                #             color_seg = color_seg[..., ::-1]

                #             # Show image + mask
                #             #img = np.array(image) * 0.5 + color_seg * 0.5
                #             #img = np.array(image_patches.cpu()) * 0.5 + color_seg * 0.5
                #             img = color_seg 
                #             img = img.astype(np.uint8)

                #             plt.figure(figsize=(15, 10))
                #             plt.imshow(img)
                #             plt.show()

                # # # Fill in the results array   
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1,2,0))
                    pred[x:x+w, y:y+h] += out
                del(outs)
                
                #break   
            
        pred = np.argmax(pred, axis=-1)

        fig = plt.figure()
        #fig.add_subplot(1,3,1)
        fig.add_subplot(1,2,1)
        plt.imshow(np.asarray(255 * img, dtype='uint8'))
        plt.axis('off')
        #fig.add_subplot(1,3,2) 
        fig.add_subplot(1,2,2)
        plt.imshow(convert_to_color(pred))
        plt.axis('off')
        #fig.add_subplot(1,3,3)
        #plt.imshow(gt)
        #plt.axis('off') 
        #plt.show() 
        plt.savefig('./theOutputImage.png')
        #plt.savefig('./tesssttt2_tttlll2.png')     
        #io.imsave('./tesssttt2_tttlll2.png') 
        #plt.pause(10)  

        all_preds.append(pred)
        all_gts.append(gt_e)
        
        #         # # prepare the image for the model 
        #         encoding = feature_extractor(img, return_tensors="pt")        
        #         pixel_values = encoding.pixel_values.to(device)

        #         img = np.array(image) * 0.5 + color_seg * 0.5
        #         img = img.astype(np.uint8)

        #         plt.figure(figsize=(15, 10))   
        #         plt.imshow(img) 
        #         plt.show() 
                
#_, all_preds, all_gts = test(model, test_ids, all=True, stride=32)       

# # save the model            
#torch.save(model.state_dict(), '/Data/ndionelis/segformermain30082023')
torch.save(model.state_dict(), '/Data/ndionelis/formodels/srmr2ehw4h44')
#model.load_state_dict(torch.load('/Data/ndionelis/segformermain30082023'))     

_, all_preds, all_gts = test(model, test_ids, all=True, stride=32)

# # # prepare the image for the model      
# encoding = feature_extractor(image, return_tensors="pt")
# pixel_values = encoding.pixel_values.to(device)

# # forward pass
#outputs = model(pixel_values=pixel_values)

from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# img = np.array(image) * 0.5 + color_seg * 0.5
# img = img.astype(np.uint8)

# plt.figure(figsize=(15, 10))
# plt.imshow(img)
# plt.show()

