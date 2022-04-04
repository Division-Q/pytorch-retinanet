import os
import numpy as np
import pandas as pd

# Set up the addresses of the dataset file.
dir_dataset = 'D:\GWU\GWU Spring 2022\Research\dataset\VisDrone2019-DET-train'
# dir_dataset = 'VisDrone2019-DET-train'
dir_labels = os.path.abspath(os.path.join(dir_dataset, 'annotations'))
dir_imgs = os.path.abspath(os.path.join(dir_dataset, 'images'))
paths_imgs = sorted(os.listdir(dir_imgs))
paths_labels = sorted(os.listdir(dir_labels))

label_dict = {1: 'pedestrian',
              2: 'people',
              3: 'bicycle',
              4: 'car',
              5: 'van',
              6: 'truck',
              7: 'tricycle',
              8: 'awning-tricycle',
              9: 'bus',
              10: 'motor',
              11: 'others'
              }

l = os.listdir(dir_labels)
train = l[0:3000]
valid = l[3000:5000]
a = np.ones([3, 4])
data = []
for item in train:
    with open(os.path.join(dir_labels, item), 'r') as f:
        lines = f.readlines()

        for line in lines:
            raw = line.split(',')
            if int(raw[4]) == 1:
                path = os.path.join(dir_imgs, item.replace('.txt', '.jpg'))
                #filename = item.split('.')
                #path = dir_imgs + filename[0] + '.jpg'
                x1 = int(raw[0])
                y1 = int(raw[1])
                x2 = int(raw[2]) + x1
                y2 = int(raw[3]) + y1
                label_index = int(raw[5])
                label = label_dict[label_index]
                #data.append([path, x1, y1, x2, y2, label_index, label])
                data.append([path, x1, y1, x2, y2, label])

data = pd.DataFrame(data, columns=['img_path', 'x1', 'y1', 'x2', 'y2', 'label'])
data.to_csv('visDrone_train.csv', header=False, index=False)

a = np.ones([3, 4])
data = []
for item in valid:
    with open(os.path.join(dir_labels, item), 'r') as f:
        lines = f.readlines()

        for line in lines:
            raw = line.split(',')
            if int(raw[4]) == 1:
                path = os.path.join(dir_imgs, item.replace('.txt', '.jpg'))
                #filename = item.split('.')
                #path = dir_imgs + filename[0] + '.jpg'
                x1 = int(raw[0])
                y1 = int(raw[1])
                x2 = int(raw[2]) + x1
                y2 = int(raw[3]) + y1
                label_index = int(raw[5])
                label = label_dict[label_index]
                #data.append([path, x1, y1, x2, y2, label_index, label])
                data.append([path, x1, y1, x2, y2, label])

data = pd.DataFrame(data, columns=['img_path', 'x1', 'y1', 'x2', 'y2', 'label'])

data.to_csv('visDrone_valid.csv', header=False, index=False)
