import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

import tqdm
from itertools import cycle, chain

BATCH_SIZE = 32

lr = 0.0002
beta1 = 0.5
beta2 = 0.999

trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
 ])
'''
mnist = torchvision.datasets.MNIST(root = 'mnist/', train = True, download = True, transform = trans)
mnist_test = torchvision.datasets.MNIST(root = 'mnist/', train = False, download = True, transform = trans)
usps = torchvision.datasets.USPS(root = 'usps/', train = True, download = True, transform = trans)

mnist_enum =  cycle(enumerate(torch.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)))
mnist_test_enum =  enumerate(torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=True))
usps_enum = cycle(enumerate(torch.utils.data.DataLoader(usps, batch_size=BATCH_SIZE, shuffle=True)))


usps_test = torchvision.datasets.USPS(root = 'usps/', train = False, download = True, transform = trans)
usps_nocycle_enum = enumerate(torch.utils.data.DataLoader(usps, batch_size=BATCH_SIZE, shuffle=True))
usps_test_enum =  enumerate(torch.utils.data.DataLoader(usps_test, batch_size=BATCH_SIZE, shuffle=True))
'''

mnist = torchvision.datasets.MNIST(root = 'mnist/', train = True, download = True, transform = trans)
mnist_test = torchvision.datasets.MNIST(root = 'mnist/', train = False, download = True, transform = trans)
svhn = torchvision.datasets.SVHN(root = 'svhn/', split = 'train', download = True, transform = trans)
#svhn_test = torchvision.datasets.SVHN(root = 'svhn/', split = 'test', download = True, transform = trans)

mnist_enum =  cycle(enumerate(torch.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)))
mnist_test_enum =  enumerate(torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=True))
mnist_nocycle_enum = enumerate(torch.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True))
svhn_enum =  cycle(enumerate(torch.utils.data.DataLoader(svhn, batch_size=BATCH_SIZE, shuffle=True)))
#svhn_test_enum =  enumerate(torch.utils.data.DataLoader(svhn_test, batch_size=BATCH_SIZE, shuffle=True))

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

class G12(nn.Module):
    """Generator for transfering from mnist to usps"""
    def __init__(self, conv_dim=64):
        super(G12, self).__init__()
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
        out = F.tanh(self.deconv2(out))              # (?, 3, 32, 32)
        return out
    
class G21(nn.Module):
    """Generator for transfering from usps to mnist"""
    def __init__(self, conv_dim=64):
        super(G21, self).__init__()
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
        out = F.tanh(self.deconv2(out))              # (?, 1, 32, 32)
        return out
    
class D1(nn.Module):
    """Discriminator for mnist."""
    def __init__(self, conv_dim=64, use_labels=False):
        super(D1, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim*4, 11, 4, 1, 0, False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out

class D2(nn.Module):
    """Discriminator for usps."""
    def __init__(self, conv_dim=64, use_labels=False):
        super(D2, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim*4, n_out, 4, 1, 0, False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out
        
class C2(nn.Module):
    """Classifier for usps."""
    def __init__(self, conv_dim=64, use_labels=False):
        super(C2, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        n_out = 11 if use_labels else 1
        self.fc = conv(conv_dim*4, 10, 4, 1, 0, False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out

g12 = G12()
g21 = G21()
d1 = D1()
d2 = D2()
c2 = C2()

celoss = nn.CrossEntropyLoss()
bceloss = nn.BCELoss()
kldloss = torch.nn.KLDivLoss()

g_params = list(g12.parameters()) + list(g21.parameters()) + list(c2.parameters())
d_params = list(d1.parameters()) + list(d2.parameters())

g_opt = optim.Adam(g_params, lr, [beta1, beta2])
d_opt = optim.Adam(d_params, lr, [beta1, beta2])

img_list = []
j = 0
for _ in range(1000):
    source_accs = []
    target_accs = []
    pbar = tqdm.tqdm(range(1000))
    for i in pbar:
        _, (real_source_x, true_source_y) = next(svhn_enum)
        _, (real_target_x, true_target_y) = next(mnist_enum)
        target_size = real_target_x.shape[0]
        fake_labels =  torch.autograd.Variable(torch.Tensor([10]*target_size).long())
        res = ""
        
        d_opt.zero_grad()
        real_source_conf = d1(real_source_x)
        real_target_conf = d2(real_target_x)
        loss = celoss(real_source_conf, true_source_y) + torch.mean((real_target_conf - 1) ** 2)
        loss.backward()
        d_opt.step()
        
        res += ' REAL LOSS: ' + str(float(loss))
        
        d_opt.zero_grad()
        fake_source_conf = d1(g21(real_target_x))
        fake_target_conf = d2(g12(real_source_x))
        loss = celoss(fake_source_conf, fake_labels) + torch.mean((fake_target_conf) ** 2)
        loss.backward()
        d_opt.step()
        
        res += ' FAKE LOSS: ' + str(float(loss))

        
        g_opt.zero_grad()
        fake_target_x = g12(real_source_x)
        fake_target_c = c2(fake_target_x)
        rec_source_x = g21(fake_target_x)
        fake_target_conf = d2(fake_target_x)
        rec_loss = torch.mean((real_source_x - rec_source_x) ** 2)
        loss = rec_loss + torch.mean((fake_target_conf - 1) ** 2) + celoss(fake_target_c, true_source_y)
        loss.backward()
        g_opt.step()
        
        res += ' REC LOSS: ' + str(float(rec_loss))
        
        if j % 500 == 0:
            img_list.append(vutils.make_grid(fake_target_x, padding=2, normalize=True))
            plt.subplot(1,2,2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imsave('results/im_usps' + str(j // 100) + '.png', np.transpose(img_list[-1].detach().numpy(),(1,2,0)))
        
        g_opt.zero_grad()
        real_target_c = torch.softmax(c2(real_target_x), 1)
        fake_source_x = g21(real_target_x)
        rec_target_x = g12(fake_source_x)
        fake_source_conf = torch.softmax(d1(fake_source_x), 1)[:, :10]
        rec_loss = torch.mean((real_target_x - rec_target_x) ** 2)
        loss = rec_loss + kldloss(real_target_c, fake_source_conf)
        loss.backward()
        g_opt.step()
        

        real_target_c = c2(real_target_x)
        loss = celoss(real_target_c, true_target_y)
        max_values = torch.argmax(real_target_c, 1)
        acc = torch.sum(max_values == true_target_y) / max_values.shape[0]
        res += ' C LOSS: ' + str(float(loss))
        res += ' ACC: ' + str(float(acc))
        
        if j % 100 == 0:
            img_list.append(vutils.make_grid(fake_source_x, padding=2, normalize=True))
            plt.subplot(1,2,2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imsave('results/im_mnist' + str(j // 100) + '.png', np.transpose(img_list[-1].detach().numpy(),(1,2,0)))
        
        j += 1
        pbar.set_description(res)
    
    torch.save(g12, 'saved_models/g12.mod')
    torch.save(g21, 'saved_models/g21.mod')
    torch.save(d1, 'saved_models/d1.mod')
    torch.save(d2, 'saved_models/d2.mod')
    torch.save(c2, 'saved_models/c2.mod')
    
    mnist_nocycle_enum = enumerate(torch.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True))
    mnist_test_enum =  enumerate(torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=True))
    
    
    accs = []
    for _, (real_target_x, true_target_y) in tqdm.tqdm(mnist_test_enum):
        real_target_c = c2(real_target_x)
        max_values = torch.argmax(real_target_c, 1)
        acc = torch.sum(max_values == true_target_y) / max_values.shape[0]
        accs.append(acc)

    print(sum(accs) / len(accs))

    accs = []
    for _, (real_target_x, true_target_y) in tqdm.tqdm(mnist_nocycle_enum):
        real_target_c = c2(real_target_x)
        max_values = torch.argmax(real_target_c, 1)
        acc = torch.sum(max_values == true_target_y) / max_values.shape[0]
        accs.append(acc)

    print(sum(accs) / len(accs))

        

        
    