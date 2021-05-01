import json
import os
from pathlib import Path

import numpy as np
import torchvision
from PIL import Image
import torchvision
import tqdm

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from util import *


def get_ground_truths(train_img_path_list, idx, anno_data):

    classes = {
        "bus": 1,
        "traffic light": 2,
        "traffic sign": 3,
        "truck": 4,
        "car": 5,
        "rider": 6,
        "pedestrian": 7,
        "bicycle" : 8,
        "motorcycle" : 9
    }
    
    total_bboxes, total_labels = [], []
    
    for i in tqdm.tqdm(idx):
        bboxes = []
        labels = []
    
        for j in range(len(anno_data[i]["labels"])):
            if "box2d" in anno_data[i]["labels"][j]:
                xmin = anno_data[i]["labels"][j]["box2d"]["x1"]
                ymin = anno_data[i]["labels"][j]["box2d"]["y1"]
                xmax = anno_data[i]["labels"][j]["box2d"]["x2"]
                ymax = anno_data[i]["labels"][j]["box2d"]["y2"]
                bbox = [xmin, ymin, xmax, ymax]
                category = anno_data[i]["labels"][j]["category"]
                if category not in classes:
                    continue
                cls = classes[category]

                bboxes.append(bbox)
                labels.append(cls)

        total_bboxes.append(Tensor(bboxes))
        total_labels.append(Tensor(labels))

    return total_bboxes, total_labels


def _load_json(path_list_idx):
    with open(path_list_idx, "r") as file:
        data = json.load(file)
    print(len(data))
    return data


class BDD(torch.utils.data.Dataset):
    def __init__(
        self, img_path, idx, anno_json_path, train
    ):  # total_bboxes_list,total_labels_list,transforms=None):
        super(BDD, self).__init__()
        self.img_path = img_path
        self.idx = idx
        self.anno_data = _load_json(anno_json_path)
        self.total_bboxes_list, self.total_labels_list = get_ground_truths(
            self.img_path, self.idx, self.anno_data
        )
        print(len(self.img_path), len(self.total_labels_list), len(self.total_labels_list))
        self.split = 'TRAIN' if train else 'TEST'
        self.classes = {
                "bus": 1,
                "traffic light": 2,
                "traffic sign": 3,
                "truck": 4,
                "car": 5,
                "rider": 6,
                "pedestrian": 7,
                "bicycle" : 8,
                "motorcycle" : 9
            }

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        img = Image.open(img_path).convert("RGB")

        labels = self.total_labels_list[idx]
        bboxes = self.total_bboxes_list[idx]
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        boxes = torch.Tensor(bboxes)
        labels = torch.Tensor(labels).long()

        img, boxes, labels = transform(img, boxes, labels, split=self.split)
        
        return img, boxes, labels
    
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels