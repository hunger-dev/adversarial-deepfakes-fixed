"""

Author: Andreas Rössler
"""
import os
import argparse


import torch
import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
from network.xception import xception
import math
import torchvision


def return_pytorch04_xception(pretrained=True):
    model = xception(pretrained=False)
    if pretrained:
        model.fc = model.last_linear
        del model.last_linear
        
        pos_model_paths = [
            '/content/AdversarialDeepFakes/faceforensics++_models_subset/xception/ffpp_c23.pth',
            '/data2/paarth/faceforensics++_models_subset/xception-b5690688.pth',
            '/home/shehzeen/AdversarialDeepFakes/xception-b5690688.pth',
            '/Users/paarthneekhara/Dev/DeepLearning/DeepFakes/xception-b5690688.pth',
            '/Users/shehzeensh/Research/Xception/xception-b5690688.pth'
        ]

        state_dict = None
        for model_path in pos_model_paths:
            if os.path.exists(model_path):
                state_dict = torch.load(model_path)
                break

        if state_dict is not None:
            # "model." prefix 제거 (이 부분이 현재 없음!)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    name = k[6:]  # "model." 제거 (6글자)
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            
            # FC layer 키 이름 수정
            if 'last_linear.1.weight' in new_state_dict:
                new_state_dict['fc.weight'] = new_state_dict.pop('last_linear.1.weight')
            if 'last_linear.1.bias' in new_state_dict:
                new_state_dict['fc.bias'] = new_state_dict.pop('last_linear.1.bias')
            
            # 수정된 state_dict로 로딩
            model.load_state_dict(new_state_dict, strict=False)
        
        model.last_linear = model.fc
        del model.fc
    return model



def return_pytorch04_meso():
    model = Meso4()
    return model




class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'xception':
            self.model = return_pytorch04_xception(pretrained=False)  # pretrained=False로 변경
            # FC layer를 먼저 교체
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
            # 이제 가중치 로딩
            self.model = self.load_pretrained_weights(self.model)
        elif modelchoice == 'resnet50' or modelchoice == 'resnet18':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def load_pretrained_weights(self, model):
        pos_model_paths = [
            '/content/AdversarialDeepFakes/faceforensics++_models_subset/xception/ffpp_c23.pth',
            # ... 기존 경로들
        ]
        
        state_dict = None
        for model_path in pos_model_paths:
            if os.path.exists(model_path):
                state_dict = torch.load(model_path)
                break
        
        if state_dict is not None:
            # "model." prefix 제거
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    name = k[6:]  # "model." 제거
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            
            # FC layer 키 제거 (크기가 다르므로)
            keys_to_remove = []
            for key in new_state_dict.keys():
                if 'last_linear' in key or 'fc' in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del new_state_dict[key]
            
            # 가중치 로딩 (FC layer 제외)
            model.load_state_dict(new_state_dict, strict=False)
        
        return model

    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


def model_selection(modelname, num_out_classes,
                    dropout=None):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if modelname == 'xception':
        return TransferModel(modelchoice='xception',
                             num_out_classes=num_out_classes), 299, \
               True, ['image'], None
    elif modelname == 'resnet18':
        return TransferModel(modelchoice='resnet18', dropout=dropout,
                             num_out_classes=num_out_classes), \
               224, True, ['image'], None

    elif modelname == "meso":
        print("Returning meso model")
        return return_pytorch04_meso(), 256, False, ['image'], None
    else:
        raise NotImplementedError(modelname)


if __name__ == '__main__':
    model, image_size, *_ = model_selection('resnet18', num_out_classes=2)
    print(model)
    model = model.cuda()
    from torchsummary import summary
    input_s = (3, image_size, image_size)
    print(summary(model, input_s))
