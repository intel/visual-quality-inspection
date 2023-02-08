###################################
### IMPORT LIBRARIES  #############
###################################
import os
import sys
sys.path.append(os.path.join(sys.path[0],'../..'))
sys.path.append('../frameworks.ai.transfer-learning/')
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
from torchvision.models.feature_extraction import create_feature_extractor


from tlt.datasets import dataset_factory
from tlt.models import model_factory
from tlt.utils.file_utils import download_and_extract_tar_file, download_file


from sklearn import metrics
from sklearn.decomposition import PCA

import simsiam.loader
import simsiam.builder


###################################
### SET VARIABLES  ################
###################################
print("Setting required variables \n")

model_name = 'resnet50'

sim_siam=False
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
im_size=224
LR = 0.00171842137353148
optimizer = 'SGD'



batch_size = 32
batch_size_ss = 64
layer='layer3'
pool=2
pca_thresholds=0.99


object_type = 'hazelnut'
device = "cpu"
num_workers=224


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
            normalize_tf = transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
            self.transform = transforms.Compose([transforms.Resize(tuple(self.im_size), interpolation=InterpolationMode.LANCZOS), transforms.ToTensor(), normalize_tf])
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


###################################
### SIMSIAM MODULES ###############
###################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        # global iii
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        # _epochs.append(iii)
        curr_loss = float(entries[-1].split()[-1][1:-1])
        # _loss.append(curr_loss)
        # iii += 1
        return curr_loss


    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train(train_loader, model, criterion, optimizer, epoch):
    print_freq=1
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
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

        if i % print_freq == 0:
            curr_loss = progress.display(i)
    return curr_loss


def save_checkpoint(state, is_best, filename, epochs, category):
    torch.save(state, filename)
    if is_best:
        # print("Best Is == >", filename)
        ## To Replace with Best Model
        os.rename(filename, f'checkpoint_{epochs}_{category}.pth.tar')
        ## To Keep copies of older models
        # shutil.copyfile(filename, f'checkpoint_{epochs}_{category}.pth.tar')
    return filename

def adjust_learning_rate(optimizer, init_lr, epoch,epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def main(args):

    print("Preparing Dataloader for Training Images \n")

    output_dir="."
    if args.simsiam:
        dim=1000
        pred_dim=250
        print("=> creating SIMSIAM feature extractor with the backbone of'{}'".format(model_name))
        model = simsiam.builder.SimSiam(
            models.__dict__[model_name],dim, pred_dim)

        # infer learning rate before changing batch size
        init_lr = LR * batch_size / 256

        criterion = nn.CosineSimilarity(dim=1).to('cpu')

        optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                        {'params': model.predictor.parameters(), 'fix_lr': True}]
        optimizer = torch.optim.SGD(optim_params, init_lr,momentum=0.9,weight_decay=1e-4)

        # Training Data loading code
        traindir_ss = os.path.join(args.path, args.category,'train')
        normalize = transforms.Normalize(mean=imagenet_mean,
                                         std=imagenet_std)

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        train_dataset_ss = datasets.ImageFolder(
            traindir_ss,
            simsiam.loader.TwoCropsTransform(transforms.Compose(augmentation)))

        train_sampler = None
        train_loader_ss = torch.utils.data.DataLoader(
            train_dataset_ss, batch_size=batch_size_ss, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)


        best_least_Loss = float('inf')
        is_best_ans = False
        for epoch in range(0, args.epochs):
            adjust_learning_rate(optimizer, init_lr, epoch,args.epochs)

            # train for one epoch
            curr_loss = train(train_loader_ss, model, criterion, optimizer, epoch)

            if (curr_loss < best_least_Loss):
                best_least_Loss = curr_loss
                is_best_ans = True
            
            ## Saves the Best Intermediate Checkpoints got till this step.
            # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            #         and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': model_name,
                'state_dict': model.state_dict(),
                # 'state_dict': model.encoder.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=is_best_ans, filename='checkpoint_{:04d}.pth.tar'.format(epoch), epochs=args.epochs, category=args.category)
            is_best_ans=False
        print('No. Of Epochs=', args.epochs)
        print('Batch Size =', batch_size_ss)
        # print('Training Loss =', _loss)
        # print(category)


        net = resnet50(pretrained=False)


        # create new OrderedDict that contains Only `Encoder Layers.`
        from collections import OrderedDict

        # original saved file with DataParallel
        ckpt = torch.load(f'checkpoint_{args.epochs}_{args.category}.pth.tar',map_location=torch.device('cpu'))
        state_dict = ckpt['state_dict']    # incase there are extra parameters in the model
        new_state_dict = OrderedDict()

        # for k, v in state_dict.items():
        #     if "encoder" in k:
        #         name =  k.replace("encoder.","")
        #         new_state_dict[name] = v
        # for k, v in new_state_dict.items():
        #     print(k)

        for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('encoder.') and not k.startswith('encoder.fc'):
                    # remove prefix
                    state_dict[k[len("encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

        # load params
        net.load_state_dict(state_dict, strict=False)



    else:
        print("Loading Backbone ResNet50 Model \n")
        # net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        model = model_factory.get_model(model_name=model_name, framework="pytorch", use_case='anomaly_detection')
        img_dir = os.path.join(args.path, args.category)
        dataset = dataset_factory.load_dataset(img_dir, 
                                       use_case='image_anomaly_detection', 
                                       framework="pytorch")
        dataset.preprocess(model.image_size, batch_size=batch_size, interpolation=InterpolationMode.LANCZOS)
        components = model.train(dataset, output_dir, layer_name=layer, pooling='avg', kernel_size=pool, pca_threshold=pca_thresholds)

        exit()



    net = net.to(device)
    net.eval()

    trainset = Mvtec(args.path,object_type=args.category,split='train',im_size=im_size)
    testset = Mvtec(args.path,object_type=args.category,split='test',defect_type='all',im_size=im_size)

    # trainset = torchvision.datasets.ImageFolder(root=os.path.join(args.path,args.category,'train'))
    # testset = torchvision.datasets.ImageFolder(root=os.path.join(args.path,args.category,'test'))

    # classes = trainset.getclasses()
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,num_workers=num_workers)

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
        tt=time.time()
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

def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Anomaly Detection Training and Inference on MVTEC Dataset')

    parser.add_argument('--simsiam', action='store_true', default=False,
                        help='flag to enable simsiam feature extractor')

    parser.add_argument('--epochs', action='store', type=int, default=2,
                        help='epochs to train simsiam feature extractor')

    parser.add_argument('--path', action='store', type=str, required = True, default="",
                        help='path for base dataset directory')

    parser.add_argument('--category', action='store', type=str, default='hazelnut',
                        help='category of the dataset, i.e. hazelnut')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args=args_parser()
    main(args)
