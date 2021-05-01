import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import tqdm
import itertools

BATCH_SIZE = 32

lr = 0.0002
beta1 = 0.5
beta2 = 0.999

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

nc = 1

SOURCE_CHANNELS = 1
TARGET_CHANNELS = 1

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

trans = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
 ])
 
mnist = torchvision.datasets.MNIST(root = 'mnist/', train = True, download = True, transform = trans)
mnist_test = torchvision.datasets.MNIST(root = 'mnist/', train = False, download = True, transform = trans)
usps = torchvision.datasets.USPS(root = 'usps/', train = True, download = True, transform = trans)
usps_test = torchvision.datasets.USPS(root = 'usps/', train = False, download = True, transform = trans)
svhn = torchvision.datasets.SVHN(root = 'svhn/', split = 'train', download = True, transform = trans)
svhn_test = torchvision.datasets.SVHN(root = 'svhn/', split = 'test', download = True, transform = trans)

def get_source(train, cycle):
    if train:
        data = enumerate(torch.utils.data.DataLoader(svhn, batch_size=BATCH_SIZE, shuffle=True))
    else:
        data = enumerate(torch.utils.data.DataLoader(svhn_test, batch_size=BATCH_SIZE, shuffle=True))
    if cycle:
        data = itertools.cycle(data)
    return data

def get_target(train, cycle):
    if train:
        data = enumerate(torch.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True))
    else:
        data = enumerate(torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=True))
    if cycle:
        data = itertools.cycle(data)
    return data

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


class G(nn.Module):
    def __init__(self, conv_dim=64):
        super(G, self).__init__()
        # encoding blocks
        self.emb = nn.Embedding(10, nz)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x, c):
        x = (x + self.emb(c))
        x = x.view(-1, nz, 1, 1)
        x = self.main(x)
        return x

class D(nn.Module):
    def __init__(self, conv_dim=64, use_labels=False):
        super(D, self).__init__()
        self.main = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
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
        out = self.main(x)
        return out
    

class Q(nn.Module):
    def __init__(self, conv_dim=64, use_labels=False):
        super(Q, self).__init__()
        self.conv1 = conv(1, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)
        self.fc = conv(conv_dim*4, 10, 4, 1, 0, False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)    # (?, 64, 16, 16)
        out = F.leaky_relu(self.conv2(out), 0.05)  # (?, 128, 8, 8)
        out = F.leaky_relu(self.conv3(out), 0.05)  # (?, 256, 4, 4)
        out = self.fc(out).squeeze()
        return out


celoss = nn.CrossEntropyLoss()
bceloss = nn.BCELoss()

def high(x):
    return bceloss(x, torch.ones_like(x))

def low(x):
    return bceloss(x, torch.zeros_like(x))

def compare(x, y):
    #return torch.mean(torch.abs(torch.softmax(x, 1) - torch.softmax(y, 1)))
    #return kldloss(x, y)
    mx = torch.argmax(x, 1)
    my = torch.argmax(y, 1)
    return celoss(x, my) + celoss(y, mx)

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


g = G().to(device)
d = D().to(device)
q = Q().to(device)


g_opt = optim.Adam(g.parameters(), lr, [beta1, beta2])
d_opt = optim.Adam(d.parameters(), lr, [beta1, beta2])
q_opt = optim.Adam(q.parameters(), lr, [beta1, beta2])

img_list = []
j = 0

source_enum = get_source(train = True, cycle = True)
target_enum = get_target(train = True, cycle = True)

for _ in range(1000):
    source_accs = []
    target_accs = []
    pbar = tqdm.tqdm(range(1000))
    for i in pbar:
        #_, (source_x, true_source_c) = next(source_enum)
        _, (target_x, true_target_c) = next(target_enum)
        res = ""
        #source_x = source_x.to(device)
        #true_source_c = true_source_c.to(device)
        target_x = target_x.to(device)
        true_target_c = true_target_c.to(device)
        
        d_opt.zero_grad()
        fake_target_c = torch.randint(0, 10, true_target_c.shape).to(device)
        fake_target_x = g(torch.randn(BATCH_SIZE, nz).to(device), fake_target_c)
        loss = high(d(target_x)) + low(d(fake_target_x))
        loss.backward()
        d_opt.step()
        
        res += ' DIS LOSS: ' + str(float(loss))
        
        g_opt.zero_grad()
        q_opt.zero_grad()
        fake_target_c = torch.randint(0, 10, true_target_c.shape).to(device)
        fake_target_x = g(torch.randn(BATCH_SIZE, nz).to(device), fake_target_c)
        gen_loss = high(d(fake_target_x))
        class_loss = celoss(q(fake_target_x), fake_target_c)
        loss = gen_loss + class_loss
        loss.backward()
        g_opt.step()
        q_opt.step()
        
        res += ' GEN LOSS: ' + str(float(gen_loss))
        res += ' CLASS LOSS: ' + str(float(class_loss))
        
        max_values = torch.argmax(q(target_x), 1)
        acc = torch.sum(max_values == true_target_c) / max_values.shape[0]
        res += ' ACC: ' + str(purity_score(true_target_c.cpu().detach().numpy(), max_values.detach().cpu().numpy()))
        
        if j % 100 == 0:
            img_list.append(vutils.make_grid(fake_target_x, padding=2, normalize=True))
            plt.subplot(1,2,2)
            plt.axis("off")
            plt.title("Fake Images")
            plt.imsave('results/im_mnist' + str(j // 100) + '.png', np.transpose(img_list[-1].detach().cpu().numpy(),(1,2,0)))
        
        j += 1
        pbar.set_description(res)
    
    #torch.save(model, 'saved_models/joint.mod')
    
    nocycle_enum = get_target(train = True, cycle = False)
    test_enum =  get_target(train = False, cycle = False)
    
    accs = []
    for _, (real_target_x, true_target_y) in tqdm.tqdm(test_enum):
        real_target_x = real_target_x.to(device)
        true_target_y = true_target_y.to(device)
        real_target_c = q(real_target_x)
        max_values = torch.argmax(real_target_c, 1)
        acc = purity_score(true_target_y.cpu().detach().numpy(), max_values.cpu().detach().numpy())
        accs.append(acc)

    print(sum(accs) / len(accs))

    accs = []
    for _, (real_target_x, true_target_y) in tqdm.tqdm(nocycle_enum):
        real_target_x = real_target_x.to(device)
        true_target_y = true_target_y.to(device)
        real_target_c = q(real_target_x)
        max_values = torch.argmax(real_target_c, 1)
        acc = purity_score(true_target_y.cpu().detach().numpy(), max_values.cpu().detach().numpy())
        accs.append(acc)

    print(sum(accs) / len(accs))
