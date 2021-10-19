
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torch.autograd import Variable




class MNISTDataModule(LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.img_size = opts.img_size
        self.batch_size = opts.batch_size
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                MNIST("MNIST", train=True, download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), 
                    transforms.ToTensor(), 
                    transforms.Normalize([0.5], [0.5])]),
                    ),
                batch_size=self.batch_size,
                shuffle=True)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, img_size, channels, latent_dim):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, channels):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Linear(128 * ds_size ** 2, 1)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity



class GAN(LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.latent_dim = opts.latent_dim
        self.save_hyperparameters()

        self.generator = Generator(img_size=opts.img_size, channels=opts.channels, latent_dim=opts.latent_dim)
        self.discriminator = Discriminator(img_size=opts.img_size, channels=opts.channels)
        self.adversarial_loss = nn.MSELoss()

    def forward(self, z):
        return self.generator(z)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch
        # Adversarial ground truths
        valid = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False).to(imgs.device)
        fake = Variable(torch.FloatTensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False).to(imgs.device)

        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim)))).to(imgs.device)

        real_imgs = imgs
        gen_imgs = self.generator(z)

        # train generator
        if optimizer_idx == 0:
            
            grid = torchvision.utils.make_grid(gen_imgs)
            grid_x = torchvision.utils.make_grid(imgs)
            self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
            self.logger.experiment.add_image("original_images", grid_x, self.current_epoch)

            g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)

            self.log('train/d_loss', g_loss, prog_bar=True)
            return g_loss

        else:
            real_logits = self.discriminator(real_imgs)
            fake_logits = self.discriminator(gen_imgs.detach())
            real_loss = self.adversarial_loss(real_logits, valid)
            fake_loss = self.adversarial_loss(fake_logits, fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            self.log('train/d_loss', d_loss, prog_bar=True)
            self.log('train/real_loss', real_loss, prog_bar=True)
            self.log('train/real_logits', real_logits.detach().mean(), prog_bar=True)
            self.log('train/fake_logits', fake_logits.detach().mean(), prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        lr = self.opts.lr
        b1 = self.opts.b1
        b2 = self.opts.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=1000, help="number of image channels")
    opt = parser.parse_args()
    # print(opt)
    model = GAN(opt)
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=200, progress_bar_refresh_rate=20)
    data = MNISTDataModule(opt)
    trainer.fit(model, data)