import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import math
import json
import tqdm
import time

from bdd import *
from util import *
from ssd import *


EPOCHS = 10
NUM_CLASSES = 10
root_anno_path = "bdd100k_labels_detection20"

ATTRIBUTE = 'timeofday'
SOURCE_FLAG = 'daytime'
TARGET_FLAG = 'night'

BATCH_SIZE = int(input("batch"))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()
lr = float(input("lr"))
momentum = float(input("momentum"))
weight_decay = float(input("decay"))
clipping = float(input("clipping"))
iterations = int(input("iterations"))
class_iterations = int(input("class_iterations"))
mod = int(input("model"))
max_comp = float(input("max_comp"))
min_comp = float(input("min_comp"))


root_img_path = "bdd100k_images/bdd100k/images/100k"
root_anno_path = "bdd100k_labels_detection20/bdd100k/labels/detection20"

train_img_path = root_img_path + "/train/"
val_img_path = root_img_path + "/val/"

train_anno_json_path = root_anno_path + "/det_v2_train_release.json"
val_anno_json_path = root_anno_path + "/det_v2_val_release.json"

with open(train_anno_json_path, "r") as file:
    train_data = json.load(file)
print(len(train_data))
with open(val_anno_json_path, "r") as file:
    test_data = json.load(file)
print(len(test_data))

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def make_dataset(train, flag):
    if train:
        data = train_data
        json_file = train_anno_json_path
        header = train_img_path
    else:
        data = test_data
        json_file = val_anno_json_path
        header = val_img_path
    
    img_list = []
    idx = []
    for i in tqdm.tqdm(range(len(data))):
        if data[i]['attributes'][ATTRIBUTE] == flag and data[i]['labels'] != None:
            img_list.append(header + data[i]['videoName'] + '.jpg')
            idx.append(i)
    dset = BDD(img_list, idx, json_file, train)
    return dset

source_train = make_dataset(True, SOURCE_FLAG)
source_test = make_dataset(False, SOURCE_FLAG)
target_train = make_dataset(True, TARGET_FLAG)
target_test = make_dataset(False, TARGET_FLAG)

def load(dset, sample):
    return torch.utils.data.DataLoader(dset,batch_size=BATCH_SIZE,shuffle=True, collate_fn=dset.collate_fn)

def get_model(num_classes):
    model = SSD300(num_classes)
    return model.to(device)
        
        
jm = get_model(NUM_CLASSES)

if mod >= 0:
    jm = torch.load('mcd_bdd100k-9_' + str(mod) + ".pth")

        
params = list(jm.parameters()) 
opt = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.1)

crit = MultiBoxLoss(priors_cxcy=jm.priors_cxcy).to(device)

def train(train_loader, test_loader, model, criterion, optimizer, epoch, print_freq):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    dlosses = AverageMeter()  # loss
    blosses = AverageMeter()
    tlosses = AverageMeter()
    start = time.time()

    # Batches
    test = load(test_loader, False)
    train = load(train_loader, False)
    for i, ((source_images, source_boxes, source_labels), (target_images, target_boxes, target_labels)) in enumerate(zip(train, test)):
        data_time.update(time.time() - start)
        if source_images.size(0) != target_images.size(0):
            test = load(test_loader, False)
            continue
            
        # Move to default device
        source_images = source_images.to(device)  # (batch_size (N), 3, 300, 300)
        source_boxes = [b.to(device) for b in source_boxes]
        source_labels = [l.to(device) for l in source_labels]
        
        target_images = target_images.to(device)  # (batch_size (N), 3, 300, 300)
        target_boxes = [b.to(device) for b in target_boxes]
        target_labels = [l.to(device) for l in target_labels]
        
        for _ in range(class_iterations):
            predicted_source_locs1, predicted_source_scores1, predicted_source_locs2, predicted_source_scores2 = model(source_images)  # (N, 8732, 4), (N, 8732, n_classes)
            loss = criterion(predicted_source_locs1, predicted_source_scores1, source_boxes, source_labels)  # scalar
            loss += criterion(predicted_source_locs2, predicted_source_scores2, source_boxes, source_labels)  # scalar
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            losses.update(loss.item(), source_images.size(0))
            del predicted_source_locs1, predicted_source_scores1, predicted_source_locs2, predicted_source_scores2, loss
        
        
        model.freeze("bottom", False)
        predicted_source_locs1, predicted_source_scores1, predicted_source_locs2, predicted_source_scores2 = model(source_images)  # (N, 8732, 4), (N, 8732, n_classes)
        predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2 = model(target_images)  # (N, 8732, 4), (N, 8732, n_classes)
        loss = criterion(predicted_source_locs1, predicted_source_scores1, source_boxes, source_labels)  # scalar
        loss += criterion(predicted_source_locs2, predicted_source_scores2, source_boxes, source_labels)
        bloss, dloss = criterion.discrep(predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2)
        loss -= max_comp * (dloss)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
        optimizer.step()
        model.freeze("bottom", True)
        dlosses.update(dloss.item(), target_images.size(0))
        blosses.update(bloss.item(), target_images.size(0))
        del predicted_source_locs1, predicted_source_scores1, predicted_source_locs2, predicted_source_scores2, loss
        del predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2, dloss, bloss
        
        model.freeze("top", False)
        for _ in range(iterations):
            predicted_source_locs1, predicted_source_scores1, predicted_source_locs2, predicted_source_scores2 = model(source_images)  # (N, 8732, 4), (N, 8732, n_classes)
            loss = criterion(predicted_source_locs1, predicted_source_scores1, source_boxes, source_labels)  # scalar
            loss += criterion(predicted_source_locs2, predicted_source_scores2, source_boxes, source_labels)  # scalar
            predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2 = model(target_images)  # (N, 8732, 4), (N, 8732, n_classes)
            bloss, dloss = criterion.discrep(predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2)
            loss += min_comp * (dloss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
            optimizer.step()
            del predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2, loss, bloss, dloss
        model.freeze("top", True)
        
        predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2 = model(target_images)
        loss = criterion(predicted_target_locs1, predicted_target_scores1, target_boxes, target_labels)  # scalar
        loss += criterion(predicted_target_locs2, predicted_target_scores2, target_boxes, target_labels)
        tlosses.update(loss.item(), target_images.size(0))
        del predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2, loss
        
        del source_images, source_boxes, source_labels, target_images, target_boxes, target_labels
        
        batch_time.update(time.time() - start)
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})'
                  'Source Loss {loss.val:.4f} ({loss.avg:.4f})'
                  'Discrep Loss {dloss.val:.4f} ({dloss.avg:.4f})'
                  'Box Loss {bloss.val:.4f} ({bloss.avg:.4f})'
                  'Target Loss {tloss.val:.4f} ({tloss.avg:.4f})'.format(epoch, i, min(len(train), len(test)),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses, dloss = dlosses, bloss = blosses, tloss = tlosses))
        if i % 200 == 0:
            torch.save(jm, 'mcd_bdd100k-9_' + str(epoch + mod + 1) + '.pth')
                                                      

def test(test_loader, model, criterion, epoch):
    model.eval()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    tlosses = AverageMeter()
    start = time.time()
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    
    i = 1
    with torch.no_grad():# Batches
        for (target_images, target_boxes, target_labels) in tqdm.tqdm(load(test_loader, False)):
            if i > 5:
                break
            target_images = target_images.to(device)  # (batch_size (N), 3, 300, 300)
            target_boxes = [b.to(device) for b in target_boxes]
            target_labels = [l.to(device) for l in target_labels]
            
            predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2 = model(target_images)
            
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_target_locs1, predicted_target_scores1,
                                                                                           min_score=0.01, max_overlap=0.45,
                                                                                           top_k=200)
            
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(target_boxes)
            true_labels.extend(target_labels)
            
            del predicted_target_locs1, predicted_target_scores1, predicted_target_locs2, predicted_target_scores2
            del target_images, target_boxes, target_labels
            del det_boxes_batch, det_labels_batch, det_scores_batch
            i += 1
        
    
    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)
    print(APs)
    print(mAP)
                                                              
    
for epoch in range(EPOCHS):
    train(source_train, target_train, jm, crit, opt, epoch, 1)
    torch.save(jm, 'mcd_bdd100k-9_' + str(epoch + mod + 1) + '.pth')
    test(target_test, jm, crit, epoch)
    test(source_test, jm, crit, epoch)
    jm = torch.load('mcd_bdd100k-9_' + str(epoch + mod + 1) + ".pth")
        
    