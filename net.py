import os
import torch
import random
import numpy as np
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.datasets as dset
import matplotlib.animation as animation
import torchvision.transforms as transforms


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


class NetworkStuff:
    def __init__(self,
                 file_name,
                 manual_seed=97,
                 generator=Generator,
                 discriminator=Discriminator,
                 dataroot='data/',
                 start=0,
                 num_epochs=5,
                 lr=0.0002,
                 beta1=0.5,
                 image_size=64,
                 batch_size=128,
                 nc=3,
                 ndf=64,
                 nz=100,
                 ngf=64):

        print("Random Seed: ", manual_seed)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)

        self.dataset = dset.ImageFolder(
            root=dataroot,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.num_epochs = num_epochs
        self.start = start

        self.file_name = file_name

        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf

        self.netG = generator(nz=self.nz, nc=self.nc, ngf=self.ngf).to(self.device)
        self.netG.apply(self.weights_init)

        self.netD = discriminator(ndf=self.ndf, nc=self.nc).to(self.device)
        self.netD.apply(self.weights_init)

        self.criterion = nn.BCELoss()

        self.fixed_noise = torch.randn(64, nz, 1, 1, device=self.device)

        self.real_label = 1
        self.fake_label = 0

        self.optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))

        self.img_list = []
        self.G_losses = []
        self.D_losses = []

    def train(self, load=False):

        print(f'training will compute on: {self.device}')

        iters = 0

        if load:
            self.load_data()

        for epoch in tqdm(range(self.start, self.num_epochs), initial=self.start):
            for i, data in enumerate(self.dataloader, 0):
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################

                # Train with all-real batch
                self.netD.zero_grad()

                # Format batch
                real_pics = data[0].to(self.device)
                batch_size = real_pics.size(0)
                label = torch.full((batch_size,),
                                   self.real_label,
                                   dtype=torch.float,
                                   device=self.device)

                # Forward pass real batch through D
                output = self.netD(real_pics).view(-1)

                # Calculate loss on all-real batch
                err_d_real = self.criterion(output, label)

                # Calculate gradients for D in backward pass
                err_d_real.backward()
                d_x = output.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)

                # Generate fake image batch with G
                fake_pics = self.netG(noise)
                label.fill_(self.fake_label)

                # Classify all fake batch with D
                output = self.netD(fake_pics.detach()).view(-1)

                # Calculate D's loss on the all-fake batch
                err_d_fake = self.criterion(output, label)

                # Calculate the gradients for this batch
                err_d_fake.backward()
                d_g_z1 = output.mean().item()

                # Add the gradients from the all-real and all-fake batches
                err_d = err_d_real + err_d_fake

                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################

                self.netG.zero_grad()

                # fake labels are real for generator cost
                label.fill_(self.real_label)

                # Since we just updated D, perform another forward pass of all-fake batch
                output = self.netD(fake_pics).view(-1)

                # Calculate G's loss based on this output
                err_g = self.criterion(output, label)

                # Calculate gradients for G
                err_g.backward()
                d_g_z2 = output.mean().item()

                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    tqdm.write('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                               % (epoch, self.num_epochs, i, len(self.dataloader),
                                  err_d.item(), err_g.item(), d_x, d_g_z1, d_g_z2))

                # Save Losses for plotting later
                self.G_losses.append(err_g.item())
                self.D_losses.append(err_d.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.num_epochs - 1) and (i == len(self.dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                iters += 1

            self.save_model()
            self.save_data()

    def plotter(self, show=True, file_name=None):
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()

        if file_name is None:
            file_name = self.file_name

        plt.savefig(os.path.join('pics', file_name + '.png'))

        if show:
            plt.show()

    def save_model(self, file_name=None):
        """
        Сохраняет параметры нейронной сети

        :param file_name: Имя файла для весов модели,
            если не передано, будет взято имя, переданное в конструктор класса
        """

        if file_name is None:
            file_name = self.file_name

        gen_path = 'models/' + file_name + '-generator' + '.pth'
        dis_path = 'models/' + file_name + '-discriminator' + '.pth'

        try:
            torch.save(self.netG.state_dict(), gen_path)
            torch.save(self.netD.state_dict(), dis_path)

        except BaseException as e:
            print('Can not save models.\nInfo: ', e, '\n')

    def load_model(self, file_name=None):
        """
        Загружает параметры нейронной сети

        :param file_name: Имя файла для весов модели,
            если не передано, будет взято имя, переданное в конструктор класса
        """

        if file_name is None:
            file_name = self.file_name

        gen_path = 'models/' + file_name + '-generator' + '.pth'
        dis_path = 'models/' + file_name + '-discriminator' + '.pth'

        try:
            self.netG.load_state_dict(torch.load(gen_path))
            self.netD.load_state_dict(torch.load(dis_path))

            self.netG = self.netG.to(self.device)
            self.netD = self.netD.to(self.device)

        except BaseException as e:
            print('Can not load models.\nInfo: ', e, '\n')

    def save_data(self, file_name=None):

        if file_name is None:
            file_name = self.file_name
        try:
            with open(os.path.join('histories/', file_name + '-g.pickle'), 'wb') as file:
                torch.save(self.G_losses, file)

            with open(os.path.join('histories/', file_name + '-d.pickle'), 'wb') as file:
                torch.save(self.D_losses, file)

            with open(os.path.join('histories/', file_name + '-imgs.pickle'), 'wb') as file:
                torch.save(self.img_list, file)

        except BaseException as e:
            print('Can not save data.\nInfo: ', e, '\n')

    def load_data(self, file_name=None):
        if file_name is None:
            file_name = self.file_name
        try:
            with open(os.path.join('histories/', file_name + '-g.pickle'), 'rb') as file:
                self.G_losses = torch.load(file)

        except BaseException as e:
            print('Can not load generator loses data.\nInfo: ', e, '\n')

        try:
            with open(os.path.join('histories/', file_name + '-d.pickle'), 'rb') as file:
                self.D_losses = torch.load(file)

        except BaseException as e:
            print('Can not load discriminator loses data.\nInfo: ', e, '\n')

        try:

            with open(os.path.join('histories/', file_name + '-imgs.pickle'), 'rb') as file:
                self.img_list = torch.load(file)

        except BaseException as e:
            print('Can not load images data\nInfo: ', e, '\n')

    def make_gif(self, fps=2, file_name=None):

        if self.img_list is not None and self.img_list != []:

            fig = plt.figure(figsize=(8, 8))
            plt.axis("off")

            ims = [[plt.title(f'epoch n. {index}'), plt.imshow(np.transpose(pic, (1, 2, 0)), animated=True)]
                   for index, pic in enumerate(self.img_list)]

            ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

            if file_name is None:
                file_name = self.file_name

            path = os.path.join('gifs', file_name + '.gif')

            ani.save(os.path.join(path), writer='imagemagick', fps=fps)

        else:
            print('Images list is empty', '\n')

    def show_samples(self, num=64):
        real_batch = next(iter(self.dataloader))

        # Plot real images
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(vutils.make_grid(real_batch[0][:num], padding=5, normalize=True), (1, 2, 0)))

        # Plot fake images
        fake = self.netG(self.fixed_noise).detach().cpu()

        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(
            np.transpose(vutils.make_grid(fake[:num], padding=2, normalize=True), (1, 2, 0)))

        plt.show()

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
