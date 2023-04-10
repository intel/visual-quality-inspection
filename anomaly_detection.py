###################################
### IMPORT LIBRARIES  #############
###################################
import os
import sys
sys.path.append('frameworks.ai.transfer-learning/')

import yaml
import argparse
import numpy as np
import torch 
from tqdm import tqdm

from workflows.vision_anomaly_detection.src import anomaly_detection_wl as ad

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearnex import patch_sklearn
patch_sklearn()


def inference_workflow(model, pca_kernel, dataset,config):
    features,gt = ad.get_features(model,dataset.test_loader,config['model'])
    pca_components = pca_kernel.transform(features.T)
    features_reconstructed = pca_kernel.inverse_transform(pca_components)
    fre = torch.square(features.T - features_reconstructed).reshape(features.T.shape)
    fre_score = torch.sum(fre, dim=1)  # NxCxHxW --> NxHxW
    print(fre_score.shape)
    return fre_score, gt
    
def train_workflow(dataset, config):
    model = ad.train(dataset, config)
    dataset._dataset.transform = dataset._train_transform
    features,labels = ad.get_features(model,dataset._train_loader,config['model'])
    pca_kernel = get_PCA_kernel(features,config)
    
    return model, pca_kernel
    
def get_PCA_kernel(features,config):
    features = features.numpy()
    pca_kernel = PCA(config['pca']['pca_thresholds'])
    pca_kernel.fit(features.T)
    return pca_kernel

def find_threshold(fpr,tpr,thr):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thr))
    return np.round(j_ordered[-1][1],2)

def compute_auroc(gt, scores):
    fpr_binary, tpr_binary, thres = metrics.roc_curve(gt, scores)
    threshold = find_threshold(fpr_binary, tpr_binary, thres)
    auroc = metrics.auc(fpr_binary, tpr_binary)
    return np.round(auroc*100,2), np.round(threshold,2)

def compute_accuracy(gt, scores, threshold):
    accuracy_score = metrics.accuracy_score(gt, [1 if i >= threshold else 0 for i in scores])
    return np.round(accuracy_score,2)

if __name__ == "__main__":
    """Base function for anomaly detection workload"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    dataset_config = config['dataset']
    model_config = config['model']
    
    dataset = ad.get_dataset(os.path.join(dataset_config['root_dir'],dataset_config['category_type']), 
                        dataset_config['image_size'],dataset_config['batch_size'])
    
    model, pca_kernel = train_workflow(dataset, config)
    
    inference_scores, gt = inference_workflow(model, pca_kernel, dataset,config)
    
    auroc, threshold = compute_auroc(gt,inference_scores)
    
    accuracy = compute_accuracy(gt, inference_scores, threshold)
    
    print(auroc)
    print(accuracy)
    