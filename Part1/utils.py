import os
import cv2
import torch
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
new utils
'''
def get_dataset_paths(dir_dataset):
    dir_labels = os.path.abspath(os.path.join(dir_dataset, 'annotations'))
    dir_images = os.path.abspath(os.path.join(dir_dataset, 'images'))
    paths_imgs = sorted(os.listdir(dir_images))
    paths_imgs = [os.path.join(dir_images, p) for p in paths_imgs]
    paths_labels = sorted(os.listdir(dir_labels))
    paths_labels = [os.path.join(dir_labels, p) for p in paths_labels]
    return paths_imgs, paths_labels


def create_label(annot_path):
    '''
    return a label dict from the path to annotation txt file
    '''
    label = {
        'boxes': [],
        'labels': []
    }
    # iterate over each bounding box and add it to the tensor
    with open(annot_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            raw = line.split(',')
            # only use annotations where the ground truth
            # confidence is high (as opposed to 0)
            if int(raw[4]) == 1:
                # get bounding box coordinates
                x1 = int(raw[0])
                y1 = int(raw[1])
                x2 = int(raw[2])
                y2 = int(raw[3])
                label_index = int(raw[5])
                label['boxes'].append([x1,y1,x2,y2])
                label['labels'].append(label_index)

    return label

def annotate_img(img_np, pred_dict, class_dict):
    '''
    img_np: numpy image
    pred_dict: a dictionary of boxes and labels
    class_dict: number to class name mapping
    shows annotated image
    '''
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (255,255,255)]

    # convert to numpy uint8, (W,H,C)
    img_np = img_np.astype('uint8')

    # add annotations
    boxes = pred_dict['boxes']
    labels = pred_dict['labels']

    for i, b in enumerate(boxes):
        # draw bounding box
        x1 = int(b[0])
        y1 = int(b[1])
        x2 = int(b[2]) + x1
        y2 = int(b[3]) + y1

        color = colors[int(labels[i])%len(colors)]
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)

        # draw label
        label = class_dict[int(labels[i])]
        cv2.putText(img_np, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2_imshow(img_np)
    return img_np

def create_CSV(dir_dataset, num_imgs, filename, class_dict):
    '''
    dir_dataset: directory containing images/ and annotations/
    num_imgs: number of images to add to CSV
    filename: creates csv file with filename
    class_dict: number to class name mapping
    '''

    img_paths, label_paths = get_dataset_paths(dir_dataset)
    img_paths = img_paths[:num_imgs]
    label_paths = label_paths[:num_imgs]

    data = []
    for img_path, label_path in zip(img_paths, label_paths):
        with open(label_path) as f:
            lines = f.readlines()
            for line in lines:
                raw = line.split(',')
                if int(raw[4]) == 1:
                    x1 = int(raw[0])
                    y1 = int(raw[1])
                    x2 = int(raw[2]) + x1
                    y2 = int(raw[3]) + y1
                    label_index = int(raw[5])
                    label = class_dict[label_index]
                    data.append([img_path, x1, y1, x2, y2, label])

    data = pd.DataFrame(data, columns=['img_path', 'x1', 'y1', 'x2', 'y2', 'label'])
    data.to_csv(filename, header=False, index=False)

'''
end new utils
'''

# conversion from img shape to tensor shape
def reshape_whc2cwh(img):
    w, h, c = img.shape
    img = img.reshape(c, w, h)
    return img

# conversion from tensor shape to image shape
def reshape_cwh2whc(img):
    c,w,h = img.shape
    img = img.reshape(w,h,c)
    return img

# takes path of image
# return loaded image as tensor
def load_img(path_img):
    np_img = cv2.imread(path_img)
    np_img = np.float32(np_img)
    np_img = reshape_whc2cwh(np_img)

    tensor = torch.from_numpy(np_img)
    return tensor

def cv2_imshow(image):
    # developed by Kanishke Gamagedara, udpated by MAE6292
    plt.figure(dpi=200)
    mode = len(np.shape(image))
    if mode == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif mode == 2:
        plt.imshow(image, cmap='gray')
    else:
        print('Unsupported image size')
        raise
    plt.xticks([]), plt.yticks([])
    plt.axis('off')
    plt.show()

# TODO: This function will be integrated in `dataset` class
def prep_annotations(dir_dataset):
    df_labels = pd.read_csv('../../UAS-vision2/visDrone_train_annotations.csv')

    unique_names = df_labels['img_path'].unique()
    imgs = []
    labels = []
    for name in unique_names:
        dict_label = {}
        df_img = df_labels.loc[df_labels['img_path'] == name]
        boxes = torch.tensor([df_img['x1'].tolist(), df_img['y1'].tolist(), df_img['x2'].tolist(), df_img['y2'].tolist()]).T
        dict_label['boxes'] = boxes.cuda()
        dict_label['path'] = name
        dict_label['labels'] = torch.tensor(df_img['label_index'].tolist()).cuda()
        img = cv2.imread(os.path.join(dir_dataset, name))
        img = reshape_whc2cwh(img)
        img = np.float32(img)
        imgs.append(torch.from_numpy(img).cuda())
        labels.append(dict_label)
        break

    return imgs, labels

# get n random image names in the dataset
def get_names(dir_dataset, n):
    dir_imgs = os.path.join(dir_dataset, 'images')
    all_images = os.listdir(dir_imgs)
    total_images = len(all_images)

    # pick subset random integers in range
    sample = np.random.choice(range(total_images), n, replace=False)

    sample_images = [all_images[s].split('.')[0] for s in sample]
    return sample_images

