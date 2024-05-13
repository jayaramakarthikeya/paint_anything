import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt


# Inspired by https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/datasets.py
class Monet2PhotoDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.transform = transform
        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))
        print(len(self.files_A),len(self.files_B))
        if len(self.files_A) > len(self.files_B):
            self.files_A, self.files_B = self.files_B, self.files_A
        self.new_perm()
        assert len(self.files_A) > 0, "Make sure you downloaded the images!"

    def new_perm(self):
        self.randperm = torch.randperm(len(self.files_B))[:len(self.files_A)]

    def __getitem__(self, index):
        if self.transform is not None:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
            item_B = self.transform(Image.open(self.files_B[self.randperm[index]]))
        if item_A.shape[0] != 3: 
            item_A = item_A.repeat(3, 1, 1)
        if item_B.shape[0] != 3: 
            item_B = item_B.repeat(3, 1, 1)
        if index == len(self) - 1:
            self.new_perm()
        # Old versions of PyTorch didn't support normalization for different-channeled images
        return (item_A - 0.5) * 2, (item_B - 0.5) * 2

    def __len__(self):
        return min(len(self.files_A), len(self.files_B))
    
if __name__=="__main__":
    load_shape = 128
    target_shape = 128
    transform = transforms.Compose([
        transforms.Resize(load_shape),
        transforms.RandomCrop(target_shape),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = Monet2PhotoDataset(root="./data/monet2photo",transform=transform)

    for data in dataset:
        #print(data)
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(data[0].permute((1,2,0)).numpy())
        axarr[1].imshow(data[1].permute((1,2,0)).numpy()) 
        break

    plt.show()