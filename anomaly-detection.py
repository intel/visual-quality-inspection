###################################
### IMPORT LIBRARIES  #############
###################################
import os
import sys
sys.path.append(os.path.join(sys.path[0],'../..'))
sys.path.append('frameworks.ai.transfer-learning/')
import torch
import numpy as np
import time
import pickle
import argparse
import json
from tqdm import tqdm
import pandas as pd
from PIL import Image
import builtins
import math
import random
import shutil
import warnings
warnings.filterwarnings("ignore")
from prettytable import PrettyTable
import datetime
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.utils.data
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.transforms.functional import InterpolationMode
from torchvision.models import resnet18, resnet50, efficientnet_b5
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B5_Weights
from torchvision.models.feature_extraction import create_feature_extractor,get_graph_node_names

import intel_extension_for_pytorch as ipex

from tlt.datasets import dataset_factory
from tlt.models import model_factory
from tlt.utils.file_utils import download_and_extract_tar_file, download_file


from sklearn import metrics
from sklearn.decomposition import PCA

from dataset import Mvtec, Repeat
from utils import AverageMeter, ProgressMeter
import simsiam.loader
import simsiam.builder

from cutpaste.model import ProjectionNet
from cutpaste.cutpaste import CutPasteNormal,CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn, get_cutpaste_transforms

from sklearnex import patch_sklearn
patch_sklearn()

###################################
### SET VARIABLES  ################
###################################
print("Setting required variables \n")

LR = 0.171842137353148

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

batch_size = 32
batch_size_ss = 64
layer='layer3'
pool=1
pca_thresholds=0.99
device = "cpu"


###################################
### TRAINING SIMSIAM  #############
###################################

def train_simsiam(train_loader, model, criterion, optimizer, epoch):
    print_freq=1
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, (images,_) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # if args.gpu is not None:
        images[0] = images[0].to('cpu')
        images[1] = images[1].to('cpu')

        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5        

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print(i,print_freq)
        if i % print_freq == 0:
            curr_loss = progress.display(i)
    return curr_loss

###################################
### TRAINING CUTPASTE  ############
###################################
def train_cutpaste(dataloader, model, criterion, optimizer, epoch,scheduler):
    print_freq=1
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    # model.train()
    # # model, optimizer = ipex.optimize(model, optimizer=optimizer)

    if epoch == args.freeze_resnet:
        print(epoch)
        model.unfreeze()

    end = time.time()
    for i, data in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)
    
        xs = [x.to(device) for x in data['data']]

        # zero the parameter gradients
        optimizer.zero_grad()

        xc = torch.cat(xs, axis=0)
        embeds, logits = model(xc)
        
        # calculate label
        y = torch.arange(len(xs), device=device)
        y = y.repeat_interleave(xs[0].size(0))
        loss = criterion(logits, y) 
        
        losses.update(loss.item(), len(data['data']))

        # regulize weights:
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if scheduler is not None:
            scheduler.step(epoch)
        if i % print_freq == 0:
            curr_loss = progress.display(i)
    return curr_loss

def save_checkpoint(state, is_best, filename, loss):
    if is_best:
        Path("./models/").mkdir(parents=True, exist_ok=True)
        print("Saving a new checkpoint with loss ",loss, " at path ", "./models/"+filename)
        torch.save(state, "./models/"+filename)

def adjust_learning_rate(optimizer, init_lr, epoch,epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

def load_checkpoint_weights(args,filename, feature_extractor=None):
    if args.model == 'resnet50':
        net = resnet50(pretrained=False)
    else:
        net = resnet18(pretrained=False)
    # original saved file with DataParallel
    ckpt = torch.load("./models/"+filename,map_location=torch.device('cpu'))
    state_dict = ckpt['state_dict']    # incase there are extra parameters in the model
    new_state_dict = OrderedDict()

    if feature_extractor == 'simsiam':
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith('encoder.') and not k.startswith('encoder.fc'):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    elif feature_extractor == 'cutpaste':
        for k in list(state_dict.keys()):
             # retain only encoder up to before the embedding layer
            if k.startswith('model.'):
                # remove prefix
                state_dict[k[len("model."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    else:
        head_layers = [512]*args.head_layer+[128]
        num_classes = state_dict["out.weight"].shape[0]
        print(num_classes)
        net = ProjectionNet(model_name=args.model,pretrained=False, head_layers=head_layers, num_classes=num_classes)
        train_nodes, eval_nodes = get_graph_node_names(net)
        # print(eval_nodes)
    #load params
    net.load_state_dict(state_dict, strict=False)
    return net

def get_model_from_directory(args):
    models=[]
    for filename in os.listdir(args.model_path):
        f = os.path.join(args.model_path, filename)
        # checking if it is a file, correct category and correct self-supervised technique
        if os.path.isfile(f) and args.category in os.path.basename(f) and args.cutpaste:
            models.append(f)
    return os.path.basename(max(models, key=os.path.getctime))
def prepare_torchscript_model(model):
    print("Preparing torchscript model")
    if args.dtype=='bf16':
        model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        # print("running bfloat16 evalation step\n")
    else:
        model = ipex.optimize(model, dtype=torch.float32, inplace=True)
        # print("running fp32 evalation step\n")

    x = torch.randn(args.batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)
    if args.dtype=='bf16':
        x = x.to(torch.bfloat16)
        with torch.cpu.amp.autocast(dtype=torch.bfloat16), torch.no_grad():
            model = torch.jit.trace(model, x, strict=False).eval()
    else:
        with torch.no_grad():
            model = torch.jit.trace(model, x, strict=False).eval()
    model = torch.jit.freeze(model)
    return model

def main(args):

    trainset = Mvtec(args.data,object_type=args.category,split='train',im_size=args.image_size)
    testset = Mvtec(args.data,object_type=args.category,split='test',defect_type='all',im_size=args.image_size)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.repeat:
        test_loader = torch.utils.data.DataLoader(Repeat(testset,args.repeat), batch_size=args.batch_size, shuffle=False,num_workers=args.workers)
    else:
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,num_workers=args.workers)

    if args.simsiam:
        dim=1000
        pred_dim=250
        print("=> creating SIMSIAM feature extractor with the backbone of'{}'".format(args.model))
        model = simsiam.builder.SimSiam(
            models.__dict__[args.model],dim, pred_dim)

        # infer learning rate before changing batch size
        init_lr = LR * batch_size / 256

        criterion = nn.CosineSimilarity(dim=1).to('cpu')

        optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                        {'params': model.predictor.parameters(), 'fix_lr': True}]
        optimizer = torch.optim.SGD(optim_params, init_lr,momentum=0.9,weight_decay=1e-4)

        # Training Data loading code
        traindir_ss = os.path.join(args.data, args.category,'train')

        train_dataset_ss = datasets.ImageFolder(traindir_ss, simsiam.loader.TwoCropsTransform(transforms.Compose(simsiam.loader.get_simsiam_augmentation())))
        train_sampler = None
        train_loader_ss = torch.utils.data.DataLoader(
            train_dataset_ss, batch_size=batch_size_ss, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)


        best_least_Loss = float('inf')
        is_best_ans = False
        file_name_least_loss=""
        print("Fine-tuning SIMSIAM Model on ", args.epochs, "epochs using ", len(train_loader_ss), " training images")
        model.train()
        model, optimizer = ipex.optimize(model, optimizer=optimizer)
        for epoch in range(0, args.epochs):
            adjust_learning_rate(optimizer, init_lr, epoch,args.epochs)

            # train for one epoch
            curr_loss = train_simsiam(train_loader_ss, model, criterion, optimizer, epoch)

            if (curr_loss < best_least_Loss):
                best_least_Loss = curr_loss
                is_best_ans = True
                file_name_least_loss = 'simsiam_{}_checkpoint_{:04d}.pth.tar'.format(args.category, epoch)
            
            ## Saves the Best Intermediate Checkpoints got till this step.
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': model.state_dict(),
                # 'state_dict': model.encoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best_ans, filename=file_name_least_loss , loss=best_least_Loss)
            is_best_ans=False
        print('No. Of Epochs=', args.epochs)
        print('Batch Size =', batch_size_ss)
        # print('Training Loss =', _loss)
        # print(category)

        net= load_checkpoint_weights(args,file_name_least_loss, feature_extractor='simsiam')

    elif args.cutpaste:
        if len(args.model_path) == 0:
            print("=> creating CUT-PASTE feature extractor with the backbone of'{}'".format(args.model))

            weight_decay = 0.00003
            momentum = 0.9

            variant_map = {'normal':CutPasteNormal, 'scar':CutPasteScar, '3way':CutPaste3Way, 'union':CutPasteUnion}
            variant = variant_map[args.cutpaste_type]

            #augmentation:
            min_scale = 1

            train_data = Mvtec(args.data, args.category, split='train',im_size=int(args.image_size * (1/min_scale)),transform = get_cutpaste_transforms(args.image_size,variant))
            dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, drop_last=False,
                                    shuffle=True, num_workers=args.workers,# collate_fn=cut_paste_collate_fn,
                                    persistent_workers=True, pin_memory=True, prefetch_factor=5)

            # create Model:
            head_layers = [512]*args.head_layer+[128]
            num_classes = 2 if variant is not CutPaste3Way else 3
            model = ProjectionNet(model_name=args.model,pretrained=True, head_layers=head_layers, num_classes=num_classes)
            model.to(device)

            if args.freeze_resnet > 0:
                model.freeze_resnet()

            criterion = torch.nn.CrossEntropyLoss()
            if args.optim == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=momentum,  weight_decay=weight_decay)
                scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epochs)
                #scheduler = None
            elif args.optim == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=weight_decay)
                scheduler = None
            else:
                print(f"ERROR unkown optimizer: {optim_name}")

            num_batches = len(dataloader)
           
            best_least_Loss = float('inf')
            is_best_ans = False
            file_name_least_loss = ""
            print("Fine-tuning CUT-PASTE Model on ", args.epochs, "epochs using ", len(train_data), " training images")
            model.train()
            model, optimizer = ipex.optimize(model, optimizer=optimizer)
            for step in range(args.epochs):
                epoch = int(step / 1)
                
                curr_loss = train_cutpaste(dataloader, model, criterion, optimizer, epoch,scheduler)

                if (curr_loss < best_least_Loss):
                    best_least_Loss = curr_loss
                    is_best_ans = True
                    file_name_least_loss = 'cutpaste_{}_checkpoint_{:04d}.pth.tar'.format(args.category, step)
                
                ## Saves the Best Intermediate Checkpoints got till this step.
                save_checkpoint({
                    'epoch': step + 1,
                    'arch': args.model,
                    'state_dict': model.state_dict(),
                    # 'state_dict': model.encoder.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best_ans, filename=file_name_least_loss, loss=best_least_Loss)
                is_best_ans=False
            net= load_checkpoint_weights(args,file_name_least_loss, feature_extractor='cutpaste')
            # temp_evaluate(train_loader,test_loader,trainset,net)
        else:
            model_path = get_model_from_directory(args)
            net= load_checkpoint_weights(args,model_path,feature_extractor='cutpaste')

    else:
        print("Loading Backbone ResNet50 Model \n")
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model = model_factory.get_model(model_name=args.model, framework="pytorch", use_case='anomaly_detection')
        img_dir = os.path.join(args.data, args.category)
        dataset = dataset_factory.load_dataset(img_dir, 
                                       use_case='image_anomaly_detection', 
                                       framework="pytorch")
        dataset.preprocess(model.image_size, batch_size=args.batch_size, interpolation=InterpolationMode.LANCZOS)
        components,auc = model.train(dataset,'.', layer_name=layer, pooling='avg', kernel_size=pool, pca_threshold=pca_thresholds)
        return len(test_loader.dataset),auc*100


    net = net.to(device)
    net.eval()


    ###################################
    ### FEATURE EXTRACTION  ###########
    ###################################
    print("Extracting features for", len(train_loader.dataset), "Training Images \n")
    eval_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
    data = next(iter(eval_loader))
    return_nodes = {l: l for l in [layer]}
    partial_model = create_feature_extractor(net, return_nodes=return_nodes)
    # partial_model = prepare_torchscript_model(partial_model)
    features = partial_model(data['data'].to(device))[layer]
    pool_out= torch.nn.functional.avg_pool2d(features, pool) if pool > 1 else features
    outputs_inner = pool_out.contiguous().view(pool_out.size(0), -1)

    data_mats_orig = torch.empty((outputs_inner.shape[1], len(trainset))).to(device)
    with torch.no_grad():
        data_idx = 0
        num_ims = 0
        tt=time.time()
        for data in tqdm(train_loader):
            images = data['data']
            labels = data['label']
            images, labels = images.to(device), labels.to(device)
            num_samples = len(labels)
            features = partial_model(images)[layer]
            pool_out= torch.nn.functional.avg_pool2d(features, pool) if pool > 1 else features
            outputs = pool_out.contiguous().view(pool_out.size(0), -1)
            oi = torch.squeeze(outputs)
            data_mats_orig[:, data_idx:data_idx+num_samples] = oi.transpose(1, 0)
            num_ims += 1
            data_idx += num_samples
    total_time=time.time()-tt

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
        for k, data in enumerate(tqdm(test_loader)):
            inputs = data['data'].contiguous(memory_format=torch.channels_last)

            labels = data['label']
            num_im = inputs.shape[0]

            features = partial_model(inputs)[layer]
            pool_out= torch.nn.functional.avg_pool2d(features, pool) if pool > 1 else features
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
        gt = gt.numpy()

    ###################################
    ### AUROC SCORE for Evaluation  ###
    ###################################
    print("AUROC is computed on",  len(test_loader.dataset), "Test Images \n")
    fpr_binary, tpr_binary, thres = metrics.roc_curve(gt, scores)
    threshold = find_threshold(fpr_binary, tpr_binary, thres)
    print("Best threshold for classification is ", threshold)
    auc_roc_binary = metrics.auc(fpr_binary, tpr_binary)
    accuracy_score = metrics.accuracy_score(gt, [1 if i>=threshold else 0 for i in scores])
    print(f'AUROC: {auc_roc_binary*100}')
    print(f'Accuracy: {accuracy_score*100}')
    return len_dataset, auc_roc_binary*100

def find_threshold(fpr,tpr,thr):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thr))
    return np.round(j_ordered[-1][1],2)

def print_datasets_results(results):
    count=1
    my_table = PrettyTable()
    my_table.field_names = ["Seq. No.","Dataset", "Test set (size)", "AUROC"]
    for result in results:
        category, len_inference_data, auroc = result[0],result[1],result[2]
        my_table.add_row([count,category.upper(), len_inference_data, np.round(auroc,2)])
        count+=1
    return my_table


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Anomaly Detection Training and Inference on MVTEC Dataset')

    parser.add_argument('--model', default="resnet50", choices=['resnet18','resnet50'], 
                        help='Backbone architecture for sim-siam and cut-paste feature extractor')

    parser.add_argument('--simsiam', action='store_true', default=False,
                        help='flag to enable simsiam feature extractor')

    parser.add_argument('--cutpaste', action='store_true', default=False,
                        help='flag to enable cut-paste feature extractor')

    parser.add_argument('--image_size', action='store', type=int, default=224,
                        help='image size')

    parser.add_argument('--epochs', action='store', type=int, default=2,
                        help='epochs to train feature extractor')

    parser.add_argument('--batch_size', action='store', type=int, default=64,
                        help='batch size for every forward opeartion')

    parser.add_argument('--optim', action='store', type=str, default='sgd',
                        help='Name of optimizer - sgd/adam')

    parser.add_argument('--data', action='store', type=str, required = True, default="",
                        help='path for base dataset directory')

    parser.add_argument('--category', action='store', type=str, default='hazelnut',
                        help='category of the dataset, i.e. hazelnut')

    parser.add_argument('--freeze_resnet', action='store',  type=int, default=20,
                        help='Epochs upto you want to freeze ResNet layers and only train the new header with FC layers')

    parser.add_argument('--cutpaste_type', default="normal", choices=['normal', 'scar', '3way', 'union'], 
                        help='cutpaste variant to use (dafault: "normal")')

    parser.add_argument('--head_layer', default=2, type=int,
                    help='number of layers in the projection head (default: 1)')
    
    parser.add_argument('--workers', default=56, type=int, help="number of workers to use for data loading (default:56)")

    parser.add_argument('--repeat', default=0, type=int, help="number of test images to use for testing (default:0)")

    parser.add_argument('--dtype', default="fp32", choices=['fp32', 'bf16'], help='data type precision of model inference (dafault: "fp32")')

    parser.add_argument('--model_path', action='store', type=str, default="",
                        help='path for feature extractor model')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=args_parser()
    d=args.data
    if args.data:
        all_categories = [os.path.join(d, o).split('/')[-1] for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
        all_categories.sort()
        if args.category == 'all':
            results=[]
            for category in all_categories:
                print("\n#### Processing "+category.upper()+ " dataset started ##########\n")
                args.category = category
                len_inference_data,auroc = main(args)
                results.append([category,len_inference_data,auroc])
                print("\n#### Processing "+category.upper()+ " dataset completed ########\n")
            print(print_datasets_results(results))
        else:
            # import pdb
            # breakpoint()
            main(args)
