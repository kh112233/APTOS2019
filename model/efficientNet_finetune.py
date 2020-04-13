from efficientnet_pytorch import EfficientNet

from torch import nn
import torch

import copy

class EfficientNetFinetune(nn.Module):

    def __init__(self, level = 'efficientnet-b0', finetune = False, test=False, pretrain_weight:str = ""):
        super(EfficientNetFinetune, self).__init__()

        """
        level:              Level of EfficientNet parameters
        finetune:           Finetune the last fully connected layer or not
        test:               Model is for test or train
        pretrain_weight:    pretrain weight for aptos data set
        """

        # Initial input parameter 
        self.level = level 
        self.finetune = finetune    
        self.test = test    

        # Get weights that pretrained on aptos 2018 dataset for efficientNet
        self.pretrain_weight = pretrain_weight

        # Get the feature extractor of efficientNet for training
        self.efficientNetModel = self.get_efficientNetModel()

        if test:
            self.load_weight()


    def forward(self, x):
        x = self.efficientNetModel(x)
        return x
    
    def get_efficientNetModel(self):

        # Load efficientNet pretrained model
        efficientNetModel = EfficientNet.from_pretrained(self.level, num_classes=1)

        # Load pretrained weight
        if(len(self.pretrain_weight) > 0 and not self.test):
            weights = torch.load(self.pretrain_weight)
            efficientNetModel.load_state_dict(weights['model'])

        # Closs gradient of feature extract if fintuning
        if(self.finetune):
            for para in efficientNetModel.parameters():
                para.requires_grad=False

        feature = efficientNetModel._fc.in_features
        efficientNetModel._fc = nn.Linear(in_features=feature,out_features=1)

        return efficientNetModel
    
    def load_weight(self):
        weights = torch.load(self.pretrain_weight)
        self.efficientNetModel.load_state_dict(weights)
    
    def save_weight(self, path):
        best_weight = copy.deepcopy(self.efficientNetModel.state_dict()) 
        torch.save(best_weight, path)
