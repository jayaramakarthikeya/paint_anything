import torch
import torch.nn as nn
import sys
sys.path.append("../")
from model_utils import NormalizationLayer, ContentLossLayer, StyleLossLayer
from typing import List
from torchsummary import summary
from torchvision import models
from dataset.finetune_utils import ImageLoader, ImageShow

class FineTuneCompiler:
    def __init__(self, baseModel:nn.Module, contentLayerNames, styleLayerNames, device='cuda:0'):
        self.baseModel = baseModel.to(device)
        self.contentLayerNames = contentLayerNames
        self.styleLayerNames = styleLayerNames
        
    
    def compile(self, contentImage, styleImage, device='cuda:0'):
        print(device)
        contentImage = contentImage.to(device)
        styleImage = styleImage.to(device)
        contentLayers=[]
        styleLayers=[]
        model = nn.Sequential()
        model.add_module('norm', NormalizationLayer())
        i = 0
        for layer in self.baseModel.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn{}'.format(i)
            model.add_module(name, layer)
            
            if name in self.contentLayerNames:
                target = model(contentImage).detach()
                layer = ContentLossLayer(target)
                model.add_module("content{}".format(i), layer)
                contentLayers.append(layer)

            if name in self.styleLayerNames:
                target = model(styleImage).detach()
                layer = StyleLossLayer(target)
                model.add_module("style{}".format(i), layer)
                styleLayers.append(layer)
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLossLayer) or isinstance(model[i], StyleLossLayer):
                break
        model = model[:(i + 1)]
        return model, contentLayers, styleLayers

if __name__ == "__main__":
    vgg19 = models.vgg19(pretrained=True).features.eval().cpu()
    contentLayerNames = ['conv4']
    styleLayerNames = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

    complier = FineTuneCompiler(vgg19,contentLayerNames=contentLayerNames,styleLayerNames=styleLayerNames,device='cpu')

    contentImage = ImageLoader("/media/karthik/DATA/cis_5190/project/data/monet2photo/testB/2014-08-01 17_41_55.jpg",size=(128,128))
    styleImage = ImageLoader("/media/karthik/DATA/cis_5190/project/data/monet2photo/trainA/00007.jpg",size=(128,128))

    ImageShow(contentImage, title='Content Image')
    ImageShow(styleImage, title='Style Image')

    model , _ , _ = complier.compile(contentImage=contentImage,styleImage=styleImage,device='cpu')
    print(model)
    #summary(model=model,input_size=(3,128,128),batch_size=1)
    vgg_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(vgg_total_params)