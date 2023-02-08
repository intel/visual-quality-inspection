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
from prettytable import PrettyTable


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

root_dir = "/DataDisk_4/pratool/dataset/"
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
        sample = {'data': image, 'label': labels, 'file_name' : img_name}

        return sample

    def getclasses(self):
        classes = [str(i) for i in range(self.num_classes)]
        c = dict()
        for i in range(len(classes)):
            c[i] = classes[i]
        return c


print("Preparing Dataloader for Training Images \n")

import time

tt=time.time()
trainset = Mvtec(root_dir,object_type=object_type,split='train',im_size=im_size)
testset = Mvtec(root_dir,object_type=object_type,split='test',defect_type='all',im_size=im_size)

classes = trainset.getclasses()
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,num_workers=num_workers)

print("Time taken in Data loader", time.time()-tt)


###################################
### LOAD MODEL  ###################
###################################
print("Loading Backbone ResNet50 Model \n")
tt=time.time()
net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
print("Time taken in Loading Model", time.time()-tt)
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
tt=time.time()
t1=0
with torch.no_grad():
    data_idx = 0
    num_ims = 0
    tt=time.time()
    for data in tqdm(train_loader):
        images = data['data']
        labels = data['label']
        images, labels = images.to(device), labels.to(device)
        num_samples = len(labels)
        t2=time.time()
        features = partial_model(images)[layer]
        pool_out=torch.nn.functional.avg_pool2d(features, pool)
        outputs = pool_out.contiguous().view(pool_out.size(0), -1)
        oi = torch.squeeze(outputs)
        data_mats_orig[:, data_idx:data_idx+num_samples] = oi.transpose(1, 0)
        t1+=time.time()-t2
        num_ims += 1
        data_idx += num_samples
total_time=time.time()-tt
print("Total Time taken in data loading + Feature Extraction", total_time)
print("Total Time taken in Feature Extraction", t1)
print("Total Time taken in data loading", total_time-t1)
###################################
### PCA MODELING  #################
###################################
print("Train PCA Kernel on training images \n")
data_mats_orig = data_mats_orig.numpy()
tt=time.time()
pca_mats = PCA(pca_thresholds)
pca_mats.fit(data_mats_orig.T)
print("Time taken in PCA training", time.time()-tt)

features_reduced = pca_mats.transform(data_mats_orig.T)
features_reconstructed = pca_mats.inverse_transform(features_reduced)


###################################
### Training Complete  ############
###################################
print("Training Complete \n")





####################################################################################
#######################INFERENCE DEMO ##############################################
####################################################################################
####################################################################################


def print_results(file_names, scores, gt, threshold=-42):
    for file_name,score,ground_truth in zip(file_names,scores, gt):
        print("File Name", file_name, "Score", score, "Ground Truth", "Good" if ground_truth==1 else "Defect", "Prediction", "Good" if score > threshold else "Defect")

def print_results_table(file_names, scores, gt, threshold=-42):
    count=1
    my_table = PrettyTable()
    my_table.field_names = ["Seq. No.","Filename", "Ground Truth", "Prediction", "Score = Defect (< "+str(threshold)+")"]
    for fns,scr,gts in zip(file_names,scores, gt):
        for f,s,g in zip(fns,scr, gts):
            s=s.item()
            g=g.item()
            path_split=f.split("/")
            my_table.add_row([count,path_split[-2]+"/"+path_split[-1], 
                "Good" if g==1 else "Defect",
                "Good" if s > threshold else "Defect",
                np.round(s,2)])
            count+=1
    print(my_table)
    return my_table


demoset = Mvtec(root_dir,object_type=object_type,split='test',defect_type='demo',im_size=im_size)
demo_loader = torch.utils.data.DataLoader(demoset, batch_size=14, shuffle=False,num_workers=num_workers)


with torch.no_grad():
    len_dataset = len(test_loader.dataset)
    gt = torch.zeros(len_dataset)
    scores = np.empty(len_dataset)
    count = 0
    fnames=[]
    scores_list=[] 
    gtruths=[]   
    for k, data in enumerate(tqdm(test_loader)):

        inputs = data['data'].to(memory_format=torch.channels_last)
        labels = data['label']
        file_names = data['file_name']
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
        scores[count: count + num_im] = -fre_score

        gt[count:count + num_im] = labels
        count += num_im
        fnames.append(file_names)
        gtruths.append(labels)
        scores_list.append(-fre_score)
    gt = gt.numpy()

print_results_table(fnames, scores_list, gtruths, threshold=-42)

###################################
### AUROC SCORE for Evaluation  ###
###################################
print("AUROC is computed on",  len(test_loader.dataset), "Test Images \n")
fpr_binary, tpr_binary, thresholds = metrics.roc_curve(gt, scores)
auc_roc_binary = metrics.auc(fpr_binary, tpr_binary)

print(f'AUROC: {auc_roc_binary*100}')


for i,j,k in zip(fpr_binary, tpr_binary, thresholds ):
    print(i,j,k)

