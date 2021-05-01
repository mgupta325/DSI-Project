import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

import tqdm
import itertools

BATCH_SIZE = 32

lr = 0.00005
beta1 = 0.5
beta2 = 0.999

SOURCE_CHANNELS = 1
TARGET_CHANNELS = 1

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
iterations = int(input("iterations"))
'''
MNIST->USPS = 0.9621/0.9851

'''

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
    """Classifier for usps."""
    def __init__(self, conv_dim=64, use_labels=False):
        super(G, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(8192, 3072)
        self.bn1_fc = nn.BatchNorm1d(3072)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=3, padding=1)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=3, padding=1)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), 8192)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        return x

class C(nn.Module):
    def __init__(self):
        super(C, self).__init__()
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
    
    def forward(self, x):
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x

class D(nn.Module):
    """Classifier for usps."""
    def __init__(self):
        super(D, self).__init__()
        self.fc2 = nn.Linear(3072, 2048)
        self.bn2_fc = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 1)
        self.bn_fc3 = nn.BatchNorm1d(1)
        
    def forward(self, x):
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x
 
g = G().to(device)
c = C().to(device)
d = D().to(device)

celoss = nn.CrossEntropyLoss()
bceloss = nn.BCEWithLogitsLoss()
kldloss = torch.nn.KLDivLoss(log_target = True)

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
    
def entropy(x):
    x = torch.softmax(x, 1)
    return -torch.mean(x * torch.log(x))

def inter(x):
    x = torch.softmax(x, 1)
    return - torch.mean(torch.mean(torch.log(x), 1))
        
c_params = list(c.parameters()) 
d_params = list(d.parameters())
g_params = list(g.parameters())

c_opt = optim.Adam(c_params, lr, [beta1, beta2])
d_opt = optim.Adam(d_params, lr, [beta1, beta2])
g_opt = optim.Adam(g_params, lr, [beta1, beta2])

img_list = []
j = 0

source_enum = get_source(train = True, cycle = True)
target_enum = get_target(train = True, cycle = True)

for _ in range(1000):
    source_accs = []
    target_accs = []
    pbar = tqdm.tqdm(range(1000))
    for i in pbar:
        _, (source_x, true_source_c) = next(source_enum)
        _, (target_x, true_target_c) = next(target_enum)
        source_x = source_x.to(device)
        true_source_c = true_source_c.to(device)
        target_x = target_x.to(device)
        true_target_c = true_target_c.to(device)
        
        res = ""
        
        c_opt.zero_grad()
        g_opt.zero_grad()
        source_c = c(g(source_x))
        class_loss = celoss(source_c, true_source_c)
        loss = class_loss
        loss.backward()
        c_opt.step()
        g_opt.step()
        
        c_opt.zero_grad()
        source_c = c(g(source_x))
        target_c = c(g(target_x))
        class_loss = celoss(source_c, true_source_c)
        entropy_loss = entropy(target_c)
        loss = class_loss - entropy_loss
        loss.backward()
        g_opt.step()
        
        
        res += ' CLASS LOSS: ' + str(float(class_loss))
        
        d_opt.zero_grad()
        source_f = g(source_x)
        target_f = g(target_x)
        loss = high(d(source_f)) + low(d(target_f))
        loss.backward()
        d_opt.step()
        
        for _ in range(iterations):
            g_opt.zero_grad()
            target_f = g(target_x)
            target_c = c(target_f)
            entropy_loss = entropy(target_c)
            gen_loss = high(d(target_f))
            loss = entropy_loss + gen_loss
            loss.backward()
            g_opt.step()
        
        res += ' GEN LOSS: ' + str(float(gen_loss))
        res += ' ENTROPY LOSS: ' + str(float(entropy_loss))
        
        max_values = torch.argmax(target_c, 1)
        acc = torch.sum(max_values == true_target_c) / max_values.shape[0]
        res += ' ACC: ' + str(float(acc))
        
        j += 1
        pbar.set_description(res)
    
    
    nocycle_enum = get_target(train = True, cycle = False)
    test_enum =  get_target(train = False, cycle = False)
    
    accs = []
    for _, (real_target_x, true_target_y) in tqdm.tqdm(test_enum):
        real_target_x = real_target_x.to(device)
        true_target_y = true_target_y.to(device)
        real_target_c = c(g(real_target_x))
        max_values = torch.argmax(real_target_c, 1)
        acc = torch.sum(max_values == true_target_y) / max_values.shape[0]
        accs.append(acc)

    print(sum(accs) / len(accs))

    accs = []
    for _, (real_target_x, true_target_y) in tqdm.tqdm(nocycle_enum):
        real_target_x = real_target_x.to(device)
        true_target_y = true_target_y.to(device)
        real_target_c = c(g(real_target_x))
        max_values = torch.argmax(real_target_c, 1)
        acc = torch.sum(max_values == true_target_y) / max_values.shape[0]
        accs.append(acc)

    print(sum(accs) / len(accs))

        

        
    