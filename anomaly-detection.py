###################################
### IMPORT LIBRARIES  #############
###################################
import os
import sys
sys.path.append(os.path.join(sys.path[0],'../..'))
import torch
import numpy as np
import time
import pickle
import argparse
import json
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torchvision.transforms as TF
from torchvision.transforms.functional import InterpolationMode
from torchvision.models import resnet18, resnet50, efficientnet_b5
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B5_Weights
from sklearn.decomposition import PCA
from sklearn import metrics
from torch.utils.data import Dataset
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
import torch.nn as nn

###################################
### SET VARIABLES  ################
###################################
print("Setting required variables \n")
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
im_size=224


batch_size = 32
layer='layer3'
pool=2
pca_thresholds=0.99

root_dir = "/data/datah/pratool/pretrained/"
object_type = 'hazelnut'
device = "cpu"
num_workers=56





###################################
### LOAD DATASET  ################
###################################

class Mvtec(Dataset):
    def __init__(self, root_dir, object_type=None, split=None, defect_type=None, im_size=None, transform=None):
        
        if split == 'train':
            # defect_type = 'good'
            csv_name = '{}_train.csv'.format(object_type)
        else:
            csv_name = '{}_{}.csv'.format(object_type, defect_type)

        csv_file = os.path.join(root_dir, object_type, csv_name)
        # self.image_folder = os.path.join(root_dir, object_type, split, defect_type)
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = os.path.join(root_dir, object_type)
        if transform:
            self.transform = transform
        else:
            self.im_size = (224, 224) if im_size is None else (im_size, im_size)
            normalize_tf = TF.Normalize(mean=imagenet_mean, std=imagenet_std)
            self.transform = TF.Compose([TF.Resize(tuple(self.im_size), interpolation=InterpolationMode.LANCZOS), TF.ToTensor(), normalize_tf])
        self.num_classes = 1


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)
        if image.mode == 'L':
            image = image.convert('RGB')
        image = self.transform(image)
        labels = self.data_frame.iloc[idx, 1]
        sample = {'data': image, 'label': labels}

        return sample

    def getclasses(self):
        classes = [str(i) for i in range(self.num_classes)]
        c = dict()
        for i in range(len(classes)):
            c[i] = classes[i]
        return c


print("Preparing Dataloader for Training Images \n")
trainset = Mvtec(root_dir,object_type=object_type,split='train',im_size=im_size)
testset = Mvtec(root_dir,object_type=object_type,split='test',defect_type='all',im_size=im_size)

classes = trainset.getclasses()
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)




###################################
### LOAD MODEL  ###################
###################################
print("Loading Backbone ResNet50 Model \n")
net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
net.eval()



###################################
### FEATURE EXTRACTION  ###########
###################################
print("Extracting features for", len(train_loader.dataset), "Training Images \n")
eval_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
data = next(iter(eval_loader))
return_nodes = {l: l for l in [layer]}
partial_model = create_feature_extractor(net, return_nodes=return_nodes)
features = partial_model(data['data'].to(device))[layer]
pool_out=torch.nn.functional.avg_pool2d(features, pool)
outputs_inner = pool_out.contiguous().view(pool_out.size(0), -1)

data_mats_orig = torch.empty((outputs_inner.shape[1], len(trainset))).to(device)
with torch.no_grad():
    data_idx = 0
    num_ims = 0
    for data in tqdm(train_loader):
        images = data['data']
        labels = data['label']
        images, labels = images.to(device), labels.to(device)
        num_samples = len(labels)
        features = partial_model(images)[layer]
        pool_out=torch.nn.functional.avg_pool2d(features, pool)
        outputs = pool_out.contiguous().view(pool_out.size(0), -1)
        oi = torch.squeeze(outputs)
        data_mats_orig[:, data_idx:data_idx+num_samples] = oi.transpose(1, 0)
        num_ims += 1
        data_idx += num_samples



###################################
### PCA MODELING  #################
###################################
print("Train PCA Kernel on training images \n")
data_mats_orig = data_mats_orig.numpy()
pca_mats = PCA(pca_thresholds)
pca_mats.fit(data_mats_orig.T)
features_reduced = pca_mats.transform(data_mats_orig.T)
features_reconstructed = pca_mats.inverse_transform(features_reduced)


###################################
### Training Complete  ############
###################################
print("Training Complete \n")



###################################
### Inference Evaluation Begins  ##
###################################
print("Inference Evaluation Begins on",  len(test_loader.dataset), "Test Images \n")


with torch.no_grad():
    len_dataset = len(test_loader.dataset)
    gt = torch.zeros(len_dataset)

    scores = np.empty(len_dataset)

    count = 0
    time_inf = 0
    time_score = 0
    for k, data in enumerate(tqdm(test_loader)):

        inputs = data['data'].to(memory_format=torch.channels_last)

        labels = data['label']
        num_im = inputs.shape[0]
        features = partial_model(inputs)[layer]
        pool_out=torch.nn.functional.avg_pool2d(features, pool)
        outputs = pool_out.contiguous().view(pool_out.size(0), -1)

        feature_shapes = outputs.shape
        oi = outputs
        oi_or = oi
        oi_j = pca_mats.transform(oi)
        oi_reconstructed = pca_mats.inverse_transform(oi_j)
        fre = torch.square(oi_or - oi_reconstructed).reshape(feature_shapes)
        fre_score = torch.sum(fre, dim=1)  # NxCxHxW --> NxHxW   
        # print(fre_map.shape)             
        # fre_score = torch.sum(fre_map, dim=(1,2))  # NxHxW --> N
        scores[count: count + num_im] = -fre_score

        gt[count:count + num_im] = labels
        count += num_im

    gt = gt.numpy()



###################################
### AUROC SCORE for Evaluation  ###
###################################
print("AUROC is computed on",  len(test_loader.dataset), "Test Images \n")
fpr_binary, tpr_binary, _ = metrics.roc_curve(gt, scores)
auc_roc_binary = metrics.auc(fpr_binary, tpr_binary)

print(f'AUROC: {auc_roc_binary*100}')