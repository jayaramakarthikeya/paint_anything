import torch
from model.unet_cyclegan import UnetGenerator
from model.resnet_cyclegan import ResnetGenerator
import matplotlib.pyplot as plt
from dataset.finetune_utils import ImageLoader
import glob


if __name__=="__main__":
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    gen_BA = UnetGenerator(input_channels=3,output_channels=3).to(device)
    save_pt = torch.load("./unet-cyclegan/cycleGAN2.pth")
    gen_BA.load_state_dict(save_pt['gen_BA'])
    gen_res = ResnetGenerator(3,3).to(device)
    save_resnet_pt = torch.load("./resnet-cyclegan/cycleGAN_resnet.pth")
    gen_res.load_state_dict(save_resnet_pt["gen_BA"])
    gen_res.eval()
    gen_BA.eval()

    N = 5
    EVAL_FOLDER = "./eval"

    eval_files = sorted(glob.glob(EVAL_FOLDER + '/*.*'))

    # Set up the figure with a Nx3 grid
    fig, axes = plt.subplots(N, 3, figsize=(10, 10), dpi=300)  # Adjust dpi as needed

    for idx, file in enumerate(eval_files):
        if idx >= 10:  # Limit to 25 images to fit in the 5x5 grid
            break
        
        img_tensor = ImageLoader(image_path=file, size=(128, 128)).to(device)
        gen_img = gen_BA(img_tensor).squeeze().permute(1, 2, 0)
        gen_res_img = gen_res(img_tensor).squeeze().permute(1,2,0)
        
        ax = axes[idx,0]  # Determine the position in the grid
        ax.imshow(img_tensor.squeeze().permute(1,2,0).cpu().numpy())
        if idx == 0:
            ax.set_title("Original Image")
        ax.axis('off')
        ax = axes[idx,1]
        if idx == 0:
            ax.set_title("Painting (U-Net)")
        ax.imshow(gen_img.detach().cpu().numpy(), interpolation='nearest', cmap='gray')
        ax.axis('off')
        ax = axes[idx,2]
        if idx == 0:
            ax.set_title("Painting (ResNet)")
        ax.imshow(gen_res_img.detach().cpu().numpy(), interpolation='nearest', cmap='gray')
        ax.axis("off")

    plt.tight_layout()
    plt.savefig('cyclegan_out.png')
