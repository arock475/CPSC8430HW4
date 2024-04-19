import torch
import torch.nn as nn
import os
import time
import numpy as np

from torch.autograd import Variable
import torch.autograd as autograd
import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 196, kernel_size=3, stride=1, padding=1)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=[196, 32, 32])
        self.leaky_relu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=[196, 16, 16])
        self.leaky_relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.layer_norm3 = nn.LayerNorm(normalized_shape=[196, 16, 16])
        self.leaky_relu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.layer_norm4 = nn.LayerNorm(normalized_shape=[196, 8, 8])
        self.leaky_relu4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.layer_norm5 = nn.LayerNorm(normalized_shape=[196, 8, 8])
        self.leaky_relu5 = nn.LeakyReLU()

        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.layer_norm6 = nn.LayerNorm(normalized_shape=[196, 8, 8])
        self.leaky_relu6 = nn.LeakyReLU()

        self.conv7 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.layer_norm7 = nn.LayerNorm(normalized_shape=[196, 8, 8])
        self.leaky_relu7 = nn.LeakyReLU()

        self.conv8 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.layer_norm8 = nn.LayerNorm(normalized_shape=[196, 4, 4])
        self.leaky_relu8 = nn.LeakyReLU()

        self.pool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer_norm1(x)
        x = self.leaky_relu1(x)

        x = self.conv2(x)
        x = self.layer_norm2(x)
        x = self.leaky_relu2(x)

        x = self.conv3(x)
        x = self.layer_norm3(x)
        x = self.leaky_relu3(x)

        x = self.conv4(x)
        x = self.layer_norm4(x)
        x = self.leaky_relu4(x)

        x = self.conv5(x)
        x = self.layer_norm5(x)
        x = self.leaky_relu5(x)

        x = self.conv6(x)
        x = self.layer_norm6(x)
        x = self.leaky_relu6(x)

        x = self.conv7(x)
        x = self.layer_norm7(x)
        x = self.leaky_relu7(x)

        x = self.conv8(x)
        x = self.layer_norm8(x)
        x = self.leaky_relu8(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        fc1_out = self.fc1(x)
        fc10_out = self.fc10(x)

        return fc1_out, fc10_out


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.lin = nn.Linear(100, 196 * 4 * 4)
        self.batch_norm0 = nn.BatchNorm1d(196 * 4 * 4)
        self.relu0 = nn.ReLU()

        self.conv1 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(196)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(196)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(196)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(196)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(196)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(196)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1)
        self.batch_norm7 = nn.BatchNorm2d(196)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(196, 3, kernel_size=3, stride=1, padding=1)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.lin(x)
        x = self.batch_norm0(x)
        x = self.relu0(x)

        x = x.view(-1, 196, 4, 4)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.batch_norm6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.batch_norm7(x)
        x = self.relu7(x)

        x = self.conv8(x)

        x = self.tanh(x)
        return x


def get_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.to(device)

    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def plot(samples):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        if i >= 25:
            break
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig


def train(trainloader):
    temp_disc_loss = []
    temp_gen_loss = []
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        X_train_batch = X_train_batch.to(device)
        Y_train_batch = Y_train_batch.to(device)
        if Y_train_batch.shape[0] < batch_size:
            continue

        if (batch_idx % gen_train_every) == 0:
            for p in discriminator.parameters():
                p.requires_grad_(False)

            generator.zero_grad()

            label = np.random.randint(0, num_classes, batch_size)
            noise = np.random.normal(0, 1, (batch_size, noise_dim))
            label_onehot = np.zeros((batch_size, num_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :num_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).to(device)
            fake_label = Variable(torch.from_numpy(label)).to(device)

            fake_data = generator(noise).to(device)
            gen_source, gen_class = discriminator(fake_data)

            gen_source = gen_source.mean()

            fake_label = fake_label.type(torch.LongTensor)
            fake_label = fake_label.to(device)
            gen_class = criterion(gen_class, fake_label)

            gen_cost = -gen_source + gen_class
            temp_gen_loss.append(gen_cost.item())
            gen_cost.backward()

            optimizer_g.step()

        for p in discriminator.parameters():
            p.requires_grad_(True)

        discriminator.zero_grad()

        label = np.random.randint(0, num_classes, batch_size)
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        label_onehot = np.zeros((batch_size, num_classes))
        label_onehot[np.arange(batch_size), label] = 1
        noise[np.arange(batch_size), :num_classes] = label_onehot[np.arange(batch_size)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise = Variable(noise).to(device)
        fake_label = Variable(torch.from_numpy(label)).to(device)
        fake_label = fake_label.type(torch.LongTensor)
        fake_label = fake_label.to(device)
        with torch.no_grad():
            fake_data = generator(noise)

        disc_fake_source, disc_fake_class = discriminator(fake_data)

        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, fake_label)

        real_data = Variable(X_train_batch).to(device)
        real_label = Variable(Y_train_batch).to(device)

        disc_real_source, disc_real_class = discriminator(real_data)

        prediction = disc_real_class.data.max(1)[1]
        accuracy = (float(prediction.eq(real_label.data).sum()) / float(batch_size)) * 100.0

        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, real_label)

        gradient_penalty = get_gradient_penalty(discriminator, real_data, fake_data)

        disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
        temp_disc_loss.append(disc_cost.item())
        disc_cost.backward()

        optimizer_d.step()
    return temp_disc_loss, temp_gen_loss


if __name__ == "__main__":
    print(f'Device: {device}')
    # base variables
    batch_size = 128
    epochs = 50
    learning_rate = 0.0002
    num_classes = 10
    noise_dim = 100
    gen_train_every = 1

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = 'wgan_checkpoints'
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)

    out_dir = 'wgan_output'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    np.random.seed(999)
    label = np.asarray(list(range(10)) * 10)
    noise = np.random.normal(0, 1, (100, noise_dim))
    label_onehot = np.zeros((100, num_classes))
    label_onehot[np.arange(100), label] = 1

    noise[np.arange(100), :num_classes] = label_onehot[np.arange(100)]
    noise = noise.astype(np.float32)

    save_noise = torch.from_numpy(noise)
    save_noise = Variable(save_noise).to(device)

    gen_loss = []
    disc_loss = []

    for epoch in range(epochs):

        generator.train()
        discriminator.train()

        time1 = time.time()
        tmp_disc, tmp_gen = train(trainloader)

        gen_loss.append(np.mean(tmp_gen))
        disc_loss.append(np.mean(tmp_disc))

        time2 = time.time()

        sec = time2 - time1
        min, sec = divmod(sec, 60)
        hr, min = divmod(min, 60)
        print(f'Epoch: {epoch} | Time: {hr:.2f} hr {min:.2f} min {sec:.2f} sec')
        with torch.no_grad():
            generator.eval()
            samples = generator(save_noise)
            samples = samples.data.cpu().numpy()
            samples += 1.0
            samples /= 2.0
            samples = samples.transpose(0, 2, 3, 1)
            generator.train()

        fig = plot(samples)
        plt.savefig(out_dir + '/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
        plt.close(fig)

        if ((epoch + 1) % 50) == 0:
            torch.save(generator, os.path.join(ckpt_dir, 'tempG_' + str(epoch) + '.model'))
            torch.save(discriminator, os.path.join(ckpt_dir, 'tempD_' + str(epoch) + '.model'))

    torch.save(generator, os.path.join(ckpt_dir, 'generator.model'))
    torch.save(discriminator, os.path.join(ckpt_dir, 'discriminator.model'))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(gen_loss, color='r')
    ax2.plot(disc_loss, color='g')

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Generator Loss")

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.set_title("Discriminator Loss")

    fig.savefig("lossfig2.png")
