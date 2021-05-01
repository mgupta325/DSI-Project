import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

import tqdm
from itertools import cycle, chain

BATCH_SIZE = 32
MNIST_SIZE = 28
SVHN_SIZE = 32

lr = 0.0002
beta1 = 0.5

trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
 ])
 
mnist = torchvision.datasets.MNIST(root = 'mnist/', train = True, download = True, transform = trans)
mnist_test = torchvision.datasets.MNIST(root = 'mnist/', train = False, download = True, transform = trans)
svhn = torchvision.datasets.SVHN(root = 'svhn/', split = 'train', download = True, transform = trans)

mnist_enum =  cycle(enumerate(torch.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)))
mnist_test_enum =  enumerate(torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=True))
svhn_enum = cycle(enumerate(torch.utils.data.DataLoader(svhn, batch_size=BATCH_SIZE, shuffle=True)))

class ClassifyMNIST(nn.Module):
    def __init__(self):
        super(ClassifyMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
class ClassifySVHN(nn.Module):
    def __init__(self):
        super(ClassifySVHN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class MNIST2SVHN(nn.Module):
    """Generator for transfering from mnist to svhn"""
    def __init__(self, conv_dim=64):
        super(MNIST2SVHN, self).__init__()
        # encoding blocks
        self.conv1 = conv(1, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        
        # residual blocks
        self.conv3 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.conv4 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        
        # decoding blocks
        self.deconv1 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 3, 4, bn=False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)      # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)    # (?, 128, 8, 8)
        
        out = F.leaky_relu(self.conv3(out), 0.05)    # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)    # ( " )
        
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.tanh(self.deconv2(out))              # (?, 3, 32, 32)
        return out
    
class SVHN2MNIST(nn.Module):
    """Generator for transfering from svhn to mnist"""
    def __init__(self, conv_dim=64):
        super(SVHN2MNIST, self).__init__()
        # encoding blocks
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        
        # residual blocks
        self.conv3 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        self.conv4 = conv(conv_dim*2, conv_dim*2, 3, 1, 1)
        
        # decoding blocks
        self.deconv1 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, 1, 4, bn=False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)      # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)    # (?, 128, 8, 8)
        
        out = F.leaky_relu(self.conv3(out), 0.05)    # ( " )
        out = F.leaky_relu(self.conv4(out), 0.05)    # ( " )
        
        out = F.leaky_relu(self.deconv1(out), 0.05)  # (?, 64, 16, 16)
        out = F.tanh(self.deconv2(out))              # (?, 1, 32, 32)
        return out
    
class DiscriminateMNIST(nn.Module):
    """Discriminator for mnist."""
    def __init__(self, conv_dim=64, use_labels=False):
        super(DiscriminateMNIST, self).__init__()
        self.conv1 = conv(2, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        n_out = 11 if use_labels else 1
        self.fc1 = conv(conv_dim*4, n_out, 4, 1, 0, False)
        self.fc2 = nn.Linear(10, 50)
        self.fc3 = nn.Linear(50, 32 * 32)
        
    def forward(self, x, digits):
        digits = F.relu(self.fc2(digits))
        digits = self.fc3(digits)
        digits = digits.view(-1, 1, 32, 32)
        x = torch.cat([x, digits], 1)

        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc1(out).squeeze()
        return out

class DiscriminateSVHN(nn.Module):
    """Discriminator for svhn."""
    def __init__(self, conv_dim=64, use_labels=False):
        super(DiscriminateSVHN, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        n_out = 11 if use_labels else 1
        self.fc1 = conv(conv_dim*4, n_out, 4, 1, 0, False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc1(out).squeeze()
        return out
    
mnist_classifier = ClassifyMNIST()
mnist_discriminator = DiscriminateMNIST()
svhn_classifier = ClassifySVHN()
svhn_discriminator = DiscriminateSVHN()
mnist_to_svhn = MNIST2SVHN()
svhn_to_mnist = SVHN2MNIST()

klloss = nn.KLDivLoss(log_target=True)
bceloss = nn.BCELoss()
celoss = nn.CrossEntropyLoss()

mnist_disc_opt = optim.Adam(mnist_discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
svhn_disc_opt = optim.Adam(svhn_discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
mnist_gen_opt = optim.Adam(chain(svhn_to_mnist.parameters(),svhn_classifier.parameters()), lr=lr, betas=(beta1, 0.999))
svhn_gen_opt = optim.Adam(mnist_to_svhn.parameters(), lr=lr, betas=(beta1, 0.999))

mnist_class_opt = optim.Adam(mnist_classifier.parameters(), lr=0.002)


losses = []
for _ in range(1000):
    pbar = tqdm.tqdm(range(1000))
    for i in pbar:
        _, (real_mnist, true_mnist_clusters) = next(mnist_enum)
        _, (real_svhn, true_svhn_clusters) = next(svhn_enum)

        fake_mnist = svhn_to_mnist(real_svhn)
        fake_svhn = mnist_to_svhn(real_mnist)
        
        real_mnist_clusters = F.one_hot(true_mnist_clusters, 10).float()
        real_svhn_clusters = svhn_classifier(real_svhn)
        
        
        real_mnist_conf = mnist_discriminator(real_mnist, real_mnist_clusters)
        fake_mnist_conf = mnist_discriminator(fake_mnist, real_svhn_clusters)
        real_svhn_conf = svhn_discriminator(real_svhn)
        fake_svhn_conf = svhn_discriminator(fake_svhn)
        
        mnist_disc_loss = -torch.mean(real_mnist_conf) + torch.mean(fake_mnist_conf)
        svhn_disc_loss = -torch.mean(real_svhn_conf) + torch.mean(fake_svhn_conf)
        
        
        res = str(float(mnist_disc_loss)) + ' ' + str(float(svhn_disc_loss))
        mnist_disc_opt.zero_grad()
        mnist_disc_loss.backward()
        mnist_disc_opt.step()
        for p in mnist_discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)

        svhn_disc_opt.zero_grad()
        svhn_disc_loss.backward()
        svhn_disc_opt.step()
        for p in svhn_discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)
        
        
        
        fake_mnist = svhn_to_mnist(real_svhn)
        fake_svhn = mnist_to_svhn(real_mnist)
        
        real_mnist_clusters = F.one_hot(true_mnist_clusters, 10).float()
        real_svhn_clusters = svhn_classifier(real_svhn)
        
        real_mnist_conf = mnist_discriminator(real_mnist, real_mnist_clusters)
        fake_mnist_conf = mnist_discriminator(fake_mnist, real_svhn_clusters)
        real_svhn_conf = svhn_discriminator(real_svhn)
        fake_svhn_conf = svhn_discriminator(fake_svhn)
        
        max_index = real_svhn_clusters.max(dim = 1)[1]
        acc = float((max_index == true_svhn_clusters).sum() / BATCH_SIZE)
        if len(losses) < 10:
            losses.append(acc)
        else:
            losses[i % 10] = acc
        av = sum(losses) / len(losses)
        res += ' ' + str(av)
        
        cycle_mnist = svhn_to_mnist(fake_svhn)
        cycle_svhn = mnist_to_svhn(fake_mnist)

        cyclic_loss = torch.mean((cycle_svhn -  real_svhn) ** 2) + torch.mean((cycle_mnist -  real_mnist) ** 2)
        mnist_gen_loss = - torch.mean(fake_mnist_conf) + cyclic_loss
        res += ' ' + str(float(cyclic_loss))
        
        
        mnist_gen_opt.zero_grad()
        mnist_gen_loss.backward()
        mnist_gen_opt.step()
        
        
        fake_mnist = svhn_to_mnist(real_svhn)
        if i % 100 == 0:
            img = fake_mnist[0, 0]
            plt.imshow(img.detach().numpy())
            plt.gray()
            plt.savefig('results' + str(i) + '.png')
            
        
        fake_svhn = mnist_to_svhn(real_mnist)
        
        real_mnist_clusters = F.one_hot(true_mnist_clusters, 10).float()
        real_svhn_clusters = svhn_classifier(real_svhn)
        
        real_mnist_conf = mnist_discriminator(real_mnist, real_mnist_clusters)
        fake_mnist_conf = mnist_discriminator(fake_mnist, real_svhn_clusters)
        real_svhn_conf = svhn_discriminator(real_svhn)
        fake_svhn_conf = svhn_discriminator(fake_svhn)

        cycle_mnist = svhn_to_mnist(fake_svhn)
        cycle_svhn = mnist_to_svhn(fake_mnist)

        cyclic_loss = torch.mean((cycle_svhn -  real_svhn) ** 2) + torch.mean((cycle_mnist -  real_mnist) ** 2)
        svhn_gen_loss = -torch.mean(fake_svhn_conf) + cyclic_loss

        svhn_gen_opt.zero_grad()
        svhn_gen_loss.backward()
        svhn_gen_opt.step()

        pbar.set_description(res)
        
    
    torch.save(mnist_discriminator.state_dict(), 'mnist_disc_opt.mod')
    torch.save(svhn_discriminator.state_dict(), 'svhn_disc_opt.mod')
    torch.save(svhn_to_mnist.state_dict(), 'mnist_gen_opt.mod')
    torch.save(mnist_to_svhn.state_dict(), 'svhn_gen_opt.mod')
    torch.save(svhn_classifier.state_dict(), 'svhn_class_opt.mod')
    torch.save(mnist_classifier.state_dict(), 'mnist_class_opt.mod')