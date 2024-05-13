import torch
from torch import nn
from torchvision import models
from dataset.finetune_utils import ImageLoader, ImageShow
from model.vgg_finetune import FineTuneCompiler

class Trainer:
    def __init__(self, model, contentLayers, styleLayers, device='cuda:0'):
        self.model = model.to(device)
        self.contentLayers = contentLayers
        self.styleLayers = styleLayers
        
    def fit(self, image, epochs=10, alpha=1, betta=1e6, device='cuda:0'):
        image = image.to(device)
        optimizer = torch.optim.LBFGS([image.requires_grad_(True)])
        content_losses=[]
        style_losses=[]
        total_losses=[]
        for epoch in range(1, epochs+1):
            def closure():
                image.clamp(0, 1)
                optimizer.zero_grad()
                self.model(image)

                content_loss=0.
                style_loss=0.

                for contentLayer in self.contentLayers:
                    content_loss += contentLayer.loss
                for styleLayer in self.styleLayers:
                    style_loss += styleLayer.loss

                loss = alpha*content_loss + betta*style_loss
                loss.backward()
                content_losses.append(alpha*content_loss.item())
                style_losses.append(betta*style_loss.item())
                total_losses.append(loss.item())
                return loss
            optimizer.step(closure)

            print('Epoch {}/{} --- Total Loss: {:.4f} '.format(epoch, epochs, total_losses[-1]))
            print('Content Loss: {:4f} --- Style Loss : {:4f}'.format(content_losses[-1], style_losses[-1]))
            print('---'*17)
        losses={'total': total_losses, 'content': content_losses, 'style': style_losses}
        return losses, torch.clamp(image, 0, 1)
    
if __name__=="__main__":
    vgg19 = models.vgg19(pretrained=True).features.eval().cpu()
    contentLayerNames = ['conv4']
    styleLayerNames = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

    compiler = FineTuneCompiler(vgg19,contentLayerNames=contentLayerNames,styleLayerNames=styleLayerNames,device='cpu')

    contentImage = ImageLoader("/media/karthik/DATA/cis_5190/project/data/monet2photo/testB/2014-08-01 17_41_55.jpg",size=(128,128))
    styleImage = ImageLoader("/media/karthik/DATA/cis_5190/project/data/monet2photo/trainA/00007.jpg",size=(128,128))

    ImageShow(contentImage, title='Content Image')
    ImageShow(styleImage, title='Style Image')

    model, contentLayers, styleLayers = compiler.compile(contentImage, styleImage, device='cpu')

    trainer = Trainer(model, contentLayers, styleLayers, device='cpu')

    inputImage = contentImage.clone()
    losses, outImage = trainer.fit(inputImage)