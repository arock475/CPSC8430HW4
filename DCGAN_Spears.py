import torch
import torch.nn as nn
import os
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.convT1 = nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.bnorm1 = nn.BatchNorm2d(512, momentum=0.9)
        self.relu = nn.ReLU(True)

        self.convT2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(256, momentum=0.9)

        self.convT3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(128, momentum=0.9)

        self.convT4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm4 = nn.BatchNorm2d(64, momentum=0.9)

        self.convT5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, noise):
        x = self.convT1(noise)
        x = self.bnorm1(x)
        x = self.relu(x)

        x = self.convT2(x)
        x = self.bnorm2(x)
        x = self.relu(x)

        x = self.convT3(x)
        x = self.bnorm3(x)
        x = self.relu(x)

        x = self.convT4(x)
        x = self.bnorm4(x)
        x = self.relu(x)

        x = self.convT5(x)
        x = self.tanh(x)
        # final size [batch_size, 3, 64, 64]

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(64, momentum=0.9)
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(128, momentum=0.9)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(256, momentum=0.9)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bnorm4 = nn.BatchNorm2d(512, momentum=0.9)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        x = self.conv1(image)
        x = self.bnorm1(x)
        x = self.leakyRelu(x)

        x = self.conv2(x)
        x = self.bnorm2(x)
        x = self.leakyRelu(x)

        x = self.conv3(x)
        x = self.bnorm3(x)
        x = self.leakyRelu(x)

        x = self.conv4(x)
        x = self.bnorm4(x)
        x = self.leakyRelu(x)

        x = self.conv5(x)
        x = self.sigmoid(x)
        return x


cudnn.benchmark = True

# set manual seed to a constant get a consistent output
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

dataset = dset.CIFAR10(root="./data", download=True,
                       transform=transforms.Compose([
                           transforms.Resize(64),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ]))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


noise_dim = 100

if __name__ == '__main__':
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
    generator = Generator().to(device)
    generator.apply(weights_init)
    print(generator)

    discriminator = Discriminator().to(device)
    discriminator.apply(weights_init)
    print(discriminator)

    criterion = nn.BCELoss()

    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, noise_dim, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    epochs = 50
    g_loss = []
    d_loss = []
    for epoch in range(epochs):
        g_loss_temp = []
        d_loss_temp = []
        for i, data in enumerate(dataloader, 0):
            # train with real
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device, dtype=torch.float)

            output = discriminator(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
            fake = generator(noise)
            output = discriminator(fake.detach()).view(-1)
            label.fill_(fake_label)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizer_d.step()

            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_g.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (
                epoch, epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            g_loss_temp.append(errG.item())
            d_loss_temp.append(errD.item())

            # save the output
            if i % 100 == 0:
                print('saving the output')
                if not os.path.exists('dcgan_output'):
                    os.makedirs('dcgan_output')
                vutils.save_image(real_cpu, 'dcgan_output/real_samples.png', normalize=True)
                fake = generator(fixed_noise)
                vutils.save_image(fake.detach(), 'dcgan_output/fake_samples_epoch_%03d.png' % (epoch), normalize=True)

        g_loss.append(np.mean(g_loss_temp))
        d_loss.append(np.mean(d_loss_temp))

        # Check pointing for every epoch
        if not os.path.exists('dcgan_checkpoints'):
            os.makedirs('dcgan_checkpoints')
        torch.save(generator.state_dict(), 'dcgan_checkpoints/netG_epoch_%d.pth' % (epoch))
        torch.save(discriminator.state_dict(), 'dcgan_checkpoints/netD_epoch_%d.pth' % (epoch))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(g_loss, color='r')
    ax2.plot(d_loss, color='g')

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Generator Loss")

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.set_title("Discriminator Loss")

    fig.savefig("lossfig.png")
