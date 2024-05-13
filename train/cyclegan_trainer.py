
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset.monet2photo_dataset import Monet2PhotoDataset
from model.unet_cyclegan import UnetGenerator 
from model.model_utils import Discriminator
from losses import get_disc_loss, get_gen_loss
from skimage import color
import numpy as np
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import show_tensor_images
torch.manual_seed(0)
plt.rcParams["figure.figsize"] = (10, 10)

class CycleGANTrainer:
    def __init__(self,dataset,generator:nn.Module,discriminator:nn.Module,n_epochs,dim_A,dim_B,display_step,batch_size,learning_rate,load_shape,target_shape) -> None:
        self.adv_criterion = nn.MSELoss() 
        self.recon_criterion = nn.L1Loss() 

        self.n_epochs = n_epochs
        self.dim_A = dim_A
        self.dim_B = dim_B
        self.display_step = display_step
        self.batch_size = batch_size
        self.lr = learning_rate
        self.load_shape = load_shape
        self.target_shape = target_shape
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # transform = transforms.Compose([
        #     transforms.Resize(self.load_shape),
        #     transforms.RandomCrop(self.target_shape),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        # ])

        self.dataset = dataset
        self.gen_AB = generator(self.dim_A, self.dim_B).to(self.device)
        self.gen_BA = generator(self.dim_B, self.dim_A).to(self.device)
        self.gen_opt = torch.optim.Adam(list(self.gen_AB.parameters()) + list(self.gen_BA.parameters()), lr=self.lr, betas=(0.5, 0.999))
        self.disc_A = discriminator(self.dim_A).to(self.device)
        self.disc_A_opt = torch.optim.Adam(self.disc_A.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.disc_B = discriminator(self.dim_B).to(self.device)
        self.disc_B_opt = torch.optim.Adam(self.disc_B.parameters(), lr=self.lr, betas=(0.5, 0.999))

        self.gen_AB = self.gen_AB.apply(self.weights_init)
        self.gen_BA = self.gen_BA.apply(self.weights_init)
        self.disc_A = self.disc_A.apply(self.weights_init)
        self.disc_B = self.disc_B.apply(self.weights_init)

    def weights_init(self,m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def train(self,save_model=False):
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        cur_step = 0

        for epoch in range(self.n_epochs):
            # Dataloader returns the batches
            for real_A, real_B in tqdm(dataloader):
                # image_width = image.shape[3]
                real_A = nn.functional.interpolate(real_A, size=self.target_shape)
                real_B = nn.functional.interpolate(real_B, size=self.target_shape)
                cur_batch_size = len(real_A)
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)

                ### Update discriminator A ###
                self.disc_A_opt.zero_grad() # Zero out the gradient before backpropagation
                with torch.no_grad():
                    fake_A = self.gen_BA(real_B)
                disc_A_loss = get_disc_loss(real_A, fake_A, self.disc_A, self.adv_criterion)
                disc_A_loss.backward(retain_graph=True) # Update gradients
                self.disc_A_opt.step() # Update optimizer

                ### Update discriminator B ###
                self.disc_B_opt.zero_grad() # Zero out the gradient before backpropagation
                with torch.no_grad():
                    fake_B = self.gen_AB(real_A)
                disc_B_loss = get_disc_loss(real_B, fake_B, self.disc_B, self.adv_criterion)
                disc_B_loss.backward(retain_graph=True) # Update gradients
                self.disc_B_opt.step() # Update optimizer

                ### Update generator ###
                self.gen_opt.zero_grad()
                gen_loss, fake_A, fake_B = get_gen_loss(
                    real_A, real_B, self.gen_AB, self.gen_BA, self.disc_A, self.disc_B, self.adv_criterion, self.recon_criterion, self.recon_criterion
                )
                gen_loss.backward() # Update gradients
                self.gen_opt.step() # Update optimizer

                # Keep track of the average discriminator loss
                mean_discriminator_loss += disc_A_loss.item() / self.display_step
                # Keep track of the average generator loss
                mean_generator_loss += gen_loss.item() / self.display_step

                ### Visualization code ###
                if cur_step % self.display_step == 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                    show_tensor_images(torch.cat([real_A, real_B]), size=(self.dim_A, self.target_shape, self.target_shape))
                    show_tensor_images(torch.cat([fake_B, fake_A]), size=(self.dim_B, self.target_shape, self.target_shape))
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0
                    # You can change save_model to True if you'd like to save the model
                    if save_model:
                        torch.save({
                            'gen_AB': self.gen_AB.state_dict(),
                            'gen_BA': self.gen_BA.state_dict(),
                            'gen_opt': self.gen_opt.state_dict(),
                            'disc_A': self.disc_A.state_dict(),
                            'disc_A_opt': self.disc_A_opt.state_dict(),
                            'disc_B': self.disc_B.state_dict(),
                            'disc_B_opt': self.disc_B_opt.state_dict()
                        }, f"./cycleGAN.pth")
                cur_step += 1


if __name__ == "__main__":
    n_epochs = 100
    dim_A = 3
    dim_B = 3
    display_step = 1000
    batch_size = 32
    lr = 0.0002
    load_shape = 128
    target_shape = 128

    transform = transforms.Compose([
            transforms.Resize(load_shape),
            transforms.RandomCrop(target_shape),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    
    dataset = Monet2PhotoDataset('./data/monet2photo',transform=transform,mode='train')

    trainer = CycleGANTrainer(dataset=dataset,generator=UnetGenerator,discriminator=Discriminator,n_epochs=n_epochs,dim_A=dim_A,dim_B=dim_B,display_step=display_step,batch_size=batch_size,learning_rate=lr,load_shape=load_shape,target_shape=target_shape)
    trainer.train(save_model=True)