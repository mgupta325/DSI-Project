import time
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle

path = r"C:\Users\madhu\Desktop\new11\imgs" # change the working directory to the path where the images are located
os.chdir(path)

# this list holds all the image filename
flowers = []

# creates a ScandirIterator aliased as files
with os.scandir(path) as files:
    # loops through each file in the directory
    for file in files:
        if file.name.endswith('.jpg'):
            # adds only the image files to the flowers list
            flowers.append(file.name)

model = VGG16()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)


def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


data = {}
p = r"save1.pkl"

# loop through each image in the dataset
for flower in flowers:

    try:
        feat = extract_features(flower, model)
        data[flower] = feat

    except:
        with open(p, 'wb') as file:
            pickle.dump(data, file)


filenames = np.array(list(data.keys()))


feat = np.array(list(data.values()))

feat = feat.reshape(-1, 4096)


# reduce the amount of dimensions in the feature vector
# pca = PCA(n_components=6, random_state=22)
# pca.fit(feat)
# x = pca.transform(feat)
x=feat
# cluster feature vectors
kmeans = KMeans(n_clusters=2, n_jobs=-1, random_state=22)
kmeans.fit(x)
# print("kmeans",kmeans.labels_)
# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames, kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

print(groups[0])#has night time images
print(groups[1]) #has day time images
# #commented code is for visualization

# function that lets you view a cluster (based on identifier)
# plt.figure()
#
# files = groups[1]+groups[0]
#     # only allow up to 30 images to be shown at a time
# if len(files) > 30:
#         print(f"Clipping cluster size from {len(files)} to 30")
#         files = files[:5]
#     # plot each image in the cluster
# for index, file in enumerate(files):
#
#         plt.subplot(2, 4, index + 1)
#         img = load_img(file)
#         img = np.array(img)
#         plt.imshow(img)
#         plt.axis('off')
# plt.show()

# plt.figure()
#
# files = groups[0]
#     # only allow up to 30 images to be shown at a time
# if len(files) > 30:
#         print(f"Clipping cluster size from {len(files)} to 30")
#         files = files[:5]
#     # plot each image in the cluster
# for index, file in enumerate(files):
#
#         plt.subplot(1, 5, index + 1)
#         img = load_img(file)
#         img = np.array(img)
#         plt.imshow(img)
#         plt.axis('off')
# plt.show()

# plt.figure()
#
# files = groups[1]
#     # only allow up to 30 images to be shown at a time
# if len(files) > 30:
#         print(f"Clipping cluster size from {len(files)} to 30")
#         files = files[:5]
#     # plot each image in the cluster
# for index, file in enumerate(files):
#
#         plt.subplot(1, 5, index + 1)
#         img = load_img(file)
#         img = np.array(img)
#         plt.imshow(img)
#         plt.axis('off')
# plt.show()





# -----------------------------------------------------------------------testing  set code--------------------

path = r"C:\Users\madhu\Desktop\new11\100k\train"
# change the working directory to the path where the testing images are located
os.chdir(path)
start=time.time()
# this list holds all the image filename
flowers1 = []

# creates a ScandirIterator aliased as files
with os.scandir(path) as files:
    # loops through each file in the directory
    for file in files:
        if file.name.endswith('.jpg'):
            # adds only the image files to the flowers list
            flowers1.append(file.name)

data1 = {}
p = r"save1.pkl"


# loop through each image in the TEST dataset
for flower in flowers1:

    try:
        feat1 = extract_features(flower, model)
        data1[flower] = feat1

    except:
        with open(p, 'wb') as file:
            pickle.dump(data, file)

filenames = np.array(list(data1.keys()))


feat1 = np.array(list(data1.values()))

feat1 = feat1.reshape(-1, 4096)

# x1 = pca.transform(feat1)

y=kmeans.predict(feat1)
# print(y)
# holds the cluster id and the images { id: [images] }
groups1 = {}
for file, cluster in zip(filenames, y):
    if cluster not in groups1.keys():
        groups1[cluster] = []
        groups1[cluster].append(file)
    else:
        groups1[cluster].append(file)
print(groups1[0]) #night time images predicted
# print(groups1[1]) #day time images predicted
end=time.time()
print(start-end)
with open("C:/Users/madhu/Desktop/new11/file.txt", "w") as f:#has night time images filename list
    for s in groups1[0]:
        f.write(s +"\n")
with open("C:/Users/madhu/Desktop/new11/file1.txt", "w") as f: #has day time images filename list
    for s in groups1[1]:
        f.write(s +"\n")
