# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

###################################
### IMPORT LIBRARIES  #############
###################################
import os
import sys
import yaml
import argparse
import numpy as np
import torch 
from tqdm import tqdm
from prettytable import PrettyTable
from dataset import Mvtec
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet18, resnet50
from sklearn import metrics
from sklearn.decomposition import PCA
import intel_extension_for_pytorch as ipex
from itertools import chain

import warnings
warnings.filterwarnings("ignore")

from sklearnex import patch_sklearn
patch_sklearn()


def get_partial_model(model,dataset, model_config):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    data = next(iter(dataloader))
    return_nodes = {l: l for l in [model_config['layer']]}
    partial_model = create_feature_extractor(model, return_nodes=return_nodes)
    features = partial_model(data['data'].to("cpu"))[model_config['layer']]
    pool_out= torch.nn.functional.avg_pool2d(features, model_config['pool']) if model_config['pool'] > 1 else features
    outputs_inner = pool_out.contiguous().view(pool_out.size(0), -1)
    return partial_model, outputs_inner.shape
    
def get_train_features(partial_model, dataset,feature_shape, config):
    print("Feature extraction for  {} training images".format(len(dataset)))
    dataset_config = config['dataset']
    model_config = config['model']
    data_mats_orig = torch.empty((feature_shape[1], len(dataset))).to("cpu")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset_config['batch_size'], shuffle=False,
                                                num_workers=config['num_workers'])
    len_dataset = len(dataloader.dataset)
    gt = torch.zeros(len_dataset)
    with torch.cpu.amp.autocast(enabled=config['precision']=='bfloat16'):
        data_idx = 0
        for data in tqdm(dataloader):
            images = data['data']
            if config['precision'] == 'bfloat16':
                images = images.to(torch.bfloat16)
            labels = data['label']
            images, labels = images.to("cpu"), labels.to("cpu")
            num_samples = len(labels)
            features = partial_model(images)[model_config['layer']]
            pool_out= torch.nn.functional.avg_pool2d(features, model_config['pool']) if model_config['pool'] > 1 else features
            outputs = pool_out.contiguous().view(pool_out.size(0), -1)
            oi = torch.squeeze(outputs)
            data_mats_orig[:, data_idx:data_idx+num_samples] = oi.transpose(1, 0)
            gt[data_idx:data_idx + num_samples] = labels
            data_idx += num_samples
        return data_mats_orig.numpy(), gt.numpy()
    
def inference_score(partial_model, pca_kernel, dataset, feature_shape, model_config):
    print("Evaluation on {} test images".format(len(dataset)))
    model_config = config['model']
    dataset_config = config['dataset']
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset_config['batch_size'], shuffle=False,
                                                num_workers=config['num_workers'])
    with torch.cpu.amp.autocast(enabled=config['precision']=='bfloat16'):
        len_dataset = len(dataset)
        gt = torch.zeros(len_dataset)
        scores = np.empty(len_dataset)
        count = 0
        img_names = []
        for k, data in enumerate(tqdm(dataloader)):
            inputs = data['data'].contiguous(memory_format=torch.channels_last)
            if config['precision'] == 'bfloat16':
                inputs = inputs.to(torch.bfloat16)
            labels = data['label']
            num_im = inputs.shape[0]

            features = partial_model(inputs)[model_config['layer']]
            pool_out= torch.nn.functional.avg_pool2d(features, model_config['pool']) if model_config['pool'] > 1 else features
            outputs = pool_out.contiguous().view(pool_out.size(0), -1)

            feature_shapes = outputs.shape
            oi = outputs
            oi_or = oi

            oi_j = pca_kernel.transform(oi)
            oi_reconstructed = pca_kernel.inverse_transform(oi_j)

            fre = torch.square(oi_or - oi_reconstructed).reshape(feature_shapes)
            fre_score = torch.sum(fre, dim=1)  # NxCxHxW --> NxHxW   
            scores[count: count + num_im] = -fre_score

            gt[count:count + num_im] = labels

            img_names.append(data['img_name'])

            count += num_im
        gt = gt.numpy()
        img_names = list(chain.from_iterable(img_names))
        return scores, gt, img_names

def get_scores(pca_kernel, features):
    count = 0
    oi_j = pca_kernel.transform(features.T)
    oi_reconstructed = pca_kernel.inverse_transform(oi_j)
    fre = torch.square(features.T - torch.tensor(oi_reconstructed))
    # print(fre.shape)
    fre_score = torch.sum(fre, dim=1)  # NxCxHxW --> NxHxW   
    # print(fre_score.shape)
    return fre_score

def inference_workflow_bk(model, pca_kernel, dataset,config):
    features,gt = ad.get_features(model,dataset.test_loader,config['model'])
    pca_components = pca_kernel.transform(features.T)
    features_reconstructed = pca_kernel.inverse_transform(pca_components)
    fre = torch.square(features.T - features_reconstructed)
    fre_score = -torch.sum(fre, dim=1)  # NxCxHxW --> NxHxW
    return fre_score, gt
    
    
def inference_workflow(model, pca_kernel, dataset,config):
    dataset._dataset.transform = dataset._validation_transform

    eval_loader = dataset.test_loader
    data_length = len(dataset.test_subset)

    print("Evaluating on {} test images".format(data_length))

    with torch.cpu.amp.autocast(enabled=config['precision']=='bfloat16'):
        gt = torch.zeros(data_length)
        scores = np.empty(data_length)
        count = 0
        partial_model = ad.get_feature_extraction_model(model,config['model']['layer'])
        model_ts = prepare_torchscript_model(partial_model, config)
        for k, (images, labels) in enumerate(tqdm(eval_loader)):
            images = images.contiguous(memory_format=torch.channels_last)
            if config['precision'] == 'bfloat16':
                images = images.to(torch.bfloat16)
            num_im = images.shape[0]
            outputs = ad.extract_features(model_ts, images, config['model']['layer'],
                                        pooling=['avg', config['model']['pool']])
            feature_shapes = outputs.shape
            oi = outputs
            oi_or = oi
            oi_j = pca_kernel.transform(oi)
            oi_reconstructed = pca_kernel.inverse_transform(oi_j)
            fre = torch.square(oi_or - oi_reconstructed).reshape(feature_shapes)
            fre_score = torch.sum(fre, dim=1)  # NxCxHxW --> NxHxW
            scores[count: count + num_im] = -fre_score
            gt[count:count + num_im] = labels
            count += num_im

        gt = gt.numpy()
    return scores, gt
    
    
def train_workflow(dataset, config):
    model = ad.train(dataset, config)
    dataset._dataset.transform = dataset._train_transform
    print("Training on {} train images".format(len(dataset.train_subset)))
    features,labels = ad.get_features(model,dataset._train_loader,config['model'])
    pca_kernel = get_PCA_kernel(features,config)
    
    return model, pca_kernel
    
def get_PCA_kernel(features,config):
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
    return np.round(accuracy_score*100,2)

def print_datasets_results(results):
    # count=1
    my_table = PrettyTable()
    my_table.field_names = ["Category", "Test set (Image count)", "AUROC", "Accuracy (%)"]
    for result in results:
        category, len_inference_data, auroc, accuracy = result[0],result[1],result[2],result[3]
        my_table.add_row([category.upper(), len_inference_data, np.round(auroc,2),accuracy])
        # count+=1
    return my_table
def load_custom_model(path, config):
    if config['model']['name'] == 'resnet50':
        model = resnet50(pretrained=False)
    else:
        model = resnet18(pretrained=False)
    try:
        path = os.path.join(path,config['model']['feature_extractor']+'_'+
                            config['model']['name']+'_'+config['dataset']['category_type']+'.pth.tar')
        if os.path.exists(path):
            ckpt = torch.load(path,map_location=torch.device('cpu'))
            print("Loading the model from the following path: {}".format(path))
        elif os.path.exists(
            os.path.join(os.getcwd(),config['tlt_wf_path'],path.replace('./',''))):
            path = os.path.join(os.getcwd(),config['tlt_wf_path'], path.replace('./',''))
            ckpt = torch.load(path,map_location=torch.device('cpu'))
            print("Loading the model from the following path: {}".format(path))
        else:
            print("Model not found, please put model in {} output directory".format(os.getcwd()))
            sys.exit()
        
        model.load_state_dict(ckpt['state_dict'] , strict=False)
        model.eval()
        return model
    except Exception as error:
        print('Error while loading custom model: ' + repr(error))
        
def prepare_torchscript_model(model, config):
    print("Preparing torchscript model in {}".format(config['precision']))
    x = torch.randn(config['dataset']['batch_size'], 3, 
                    config['dataset']['image_size'], config['dataset']['image_size']).contiguous(memory_format=torch.channels_last)
    if config['precision']=='bfloat16':
        model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        x = x.to(torch.bfloat16)
        with torch.cpu.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
            model = torch.jit.trace(model, x, strict=False).eval()
        # print("running bfloat16 evaluation step\n")
    elif config['precision']=='float32':
        model = ipex.optimize(model, dtype=torch.float32, inplace=True)
        with torch.no_grad():
            model = torch.jit.trace(model, x, strict=False).eval()
        # print("running float32 evaluation step\n")
    else:
        model = ipex.optimize(model, inplace=True)
        
    model = torch.jit.freeze(model)
    return model
def main(config):
    dataset_config = config['dataset']
    model_config = config['model']
    fine_tune = config['fine_tune']
    if fine_tune:
        global ad
        sys.path.append('frameworks.ai.transfer-learning/')
        from workflows.vision_anomaly_detection.src import anomaly_detection_wl as ad
        
        dataset = ad.get_dataset(os.path.join(dataset_config['root_dir'],dataset_config['category_type']), 
                            dataset_config['image_size'],dataset_config['batch_size'])
        
        model, pca_kernel = train_workflow(dataset, config)
        
        inference_scores, gt = inference_workflow(model, pca_kernel, dataset,config)
        
        auroc, threshold = compute_auroc(gt,inference_scores)
        
        accuracy = compute_accuracy(gt, inference_scores, threshold)
        
        print("AUROC  {} on test images".format(auroc))
        print("Accuracy {}% on test images".format(accuracy))
        return [dataset_config['category_type'],len(dataset.test_subset),auroc,accuracy]
    else:
        model = load_custom_model(config['output_path'],config)
        trainset = Mvtec(dataset_config['root_dir'],object_type=dataset_config['category_type'],split='train',
                        im_size=dataset_config['image_size'])
        testset = Mvtec(dataset_config['root_dir'],object_type=dataset_config['category_type'],split='test',
                        defect_type='all',im_size=dataset_config['image_size'])

        partial_model, feature_shape =  get_partial_model(model,trainset, model_config)
        
        model_ts = prepare_torchscript_model(partial_model, config)
        
        train_features, train_gt = get_train_features(model_ts, trainset, feature_shape, config)
        pca_kernel = get_PCA_kernel(train_features,config)
        
        scores, test_gt, img_names = inference_score(model_ts, pca_kernel, testset, feature_shape, model_config)
        auroc, threshold = compute_auroc(test_gt,scores)
        
        accuracy = compute_accuracy(test_gt, scores, threshold)
        print("Saving prediction scores in scores.csv file")
        np.savetxt('scores.csv', np.column_stack((img_names,scores,[threshold for i in range(len(scores))],
            [1 if i >= threshold else 0 for i in scores], test_gt)),
            header='image_path,pred_score, threshold, final_score, gt_score', delimiter=',',fmt='%s')
        print("Inference on {} test images are completed!!!".format(len(testset)))
        print("AUROC  {} on test images".format(auroc))
        print("Accuracy {}% on test images".format(accuracy))
        return [dataset_config['category_type'],len(testset),auroc,accuracy]
        
    
if __name__ == "__main__":
    """Base function for anomaly detection workload"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)
    root_dir = config['dataset']['root_dir']
    category = config['dataset']['category_type']
    all_categories = [os.path.join(root_dir, o).split('/')[-1] for o in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,o))]
    all_categories.sort()
    if category == 'all':
        results=[]
        for category in all_categories:
            print("\n#### Processing "+category.upper()+ " dataset started ##########\n")
            config['dataset']['category_type'] = category
            result = main(config)
            print(print_datasets_results([result]))
            print("\n#### Processing "+category.upper()+ " dataset completed ########\n")
            results.append(result)
        print(print_datasets_results(results))
    else:
        print("\n#### Processing "+category.upper()+ " dataset started ##########\n")
        results= main(config)
        print(print_datasets_results([results]))
        print("\n#### Processing "+category.upper()+ " dataset completed ########\n")
    
