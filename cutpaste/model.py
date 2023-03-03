import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
import sys

class ProjectionNet(nn.Module):
    def __init__(self, model_name = 'resnet18', pretrained=True, head_layers=[512,512,512,512,512,512,512,512,128], num_classes=2):
        super(ProjectionNet, self).__init__()
        #self.resnet18 = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=pretrained)
        if model_name == 'resnet18':
            self.model = resnet18(pretrained=pretrained)
            last_layer = 512
        elif model_name == 'resnet50':
            self.model = resnet50(pretrained=pretrained)
            last_layer = 2048
        else:
            sys.exit(f"ERROR, only supported models are resnet18 and resnet50")

        # create MPL head as seen in the code in: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
        # TODO: check if this is really the right architecture
        
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(nn.BatchNorm1d(num_neurons))
            sequential_layers.append(nn.ReLU(inplace=True))
            last_layer = num_neurons
        
        #the last layer without activation

        head = nn.Sequential(
            *sequential_layers
          )
        self.model.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes)
    
    def forward(self, x):
        embeds = self.model(x)
        tmp = self.head(embeds)
        logits = self.out(tmp)
        return embeds, logits
    
    def freeze_resnet(self):
        # freez full resnet18
        for param in self.model.parameters():
            param.requires_grad = False
        
        #unfreeze head:
        for param in self.model.fc.parameters():
            param.requires_grad = True
            
    def unfreeze(self):
        #unfreeze all:
        for param in self.parameters():
            param.requires_grad = True
