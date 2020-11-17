import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from glob import glob
import torch
import os
import shutil

from gan.utils import sample_noise, show_images, deprocess_img, preprocess_img


def train(disc, gen, d_opt, g_opt, discriminator_loss, generator_loss, show_every=250,
          batch_size=128, noise_size=100, num_epochs=10, train_loader=None, device=None):
    """
    Train loop for GAN.
    
    The loop will consist of two steps: a discriminator step and a generator step.
    
    (1) In the discriminator step, you should zero gradients in the discriminator 
    and sample noise to generate a fake data batch using the generator. Calculate 
    the discriminator output for real and fake data, and use the output to compute
    discriminator loss. Call backward() on the loss output and take an optimizer
    step for the discriminator.
    
    (2) For the generator step, you should once again zero gradients in the generator
    and sample noise to generate a fake data batch. Get the discriminator output
    for the fake data batch and use this to compute the generator loss. Once again
    call backward() on the loss and take an optimizer step.
    
    You will need to reshape the fake image tensor outputted by the generator to 
    be dimensions (batch_size x input_channels x img_size x img_size).
    
    Use the sample_noise function to sample random noise, and the discriminator_loss
    and generator_loss functions for their respective loss computations.
    
    
    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    - train_loader: image dataloader
    - device: PyTorch device
    """
    iter_count = 0
    for folder in glob('stored/epoch*'):
        shutil.rmtree(folder)
    for epoch in range(num_epochs):
        for x, _ in tqdm(train_loader, desc=f"Training {epoch + 1}/{num_epochs}", position=0, leave=False):
            _, input_channels, img_size, _ = x.shape
            real_images = preprocess_img(x).to(device)
            d_opt.zero_grad()
            noise = sample_noise(batch_size, noise_size).reshape(batch_size, noise_size, 1, 1).to(device)
            fake_images = gen(noise)
            fake, real = disc(fake_images), disc(real_images)
            d_error = discriminator_loss(real, fake)
            d_error.backward()
            d_opt.step()

            g_opt.zero_grad()
            noise = sample_noise(batch_size, noise_size).reshape(batch_size, noise_size, 1, 1).to(device)
            fake_images = gen(noise)
            fake = disc(fake_images)
            g_error = generator_loss(fake)
            g_error.backward()
            g_opt.step()

            # Logging and output visualization
            if iter_count % show_every == 0:
                print('\nIter: {}, D: {:.4}, G:{:.4}, saving...'.format(iter_count, d_error.item(), g_error.item()))
                dir_name = f'stored/epoch{epoch}_{round(d_error.item(), 3)}_{round(g_error.item(), 3)}'
                os.mkdir(dir_name)
                torch.save(gen.state_dict(), os.path.join(dir_name, 'generator.pth'))
                torch.save(disc.state_dict(), os.path.join(dir_name, 'discriminator.pth'))
                disp_fake_images = deprocess_img(fake_images.data)
                imgs_numpy = disp_fake_images.cpu().numpy()
                show_images(imgs_numpy[0:16], color=input_channels != 1)
                plt.savefig(os.path.join(dir_name, 'plot.png'))
                plt.show()

            iter_count += 1
