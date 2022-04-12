import os
import cv2
import torch
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

## get n random image names in the dataset
#def get_names(dir_dataset, n):
#    dir_imgs = os.path.join(dir_dataset, 'images')
#    all_images = os.listdir(dir_imgs)
#    total_images = len(all_images)
#
#    # pick subset random integers in range
#    sample = np.random.choice(range(total_images), n, replace=False)
#
#    sample_images = [all_images[s].split('.')[0] for s in sample]
#    return sample_images

def annotate_img(img_tensor, pred_dict):
    label_dict = {0: 'place-holder', # fix this
                  1: 'pedestrian',
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
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (255,255,255)]

    # convert to numpy uint8, (W,H,C)
    img_np = img_tensor.cpu().numpy()
    img_np = img_np.astype('uint8')
    img_np = reshape_cwh2whc(img_np)

    # add annotations
    boxes = pred_dict['boxes'].cpu()
    labels = pred_dict['labels'].cpu()
    #print(labels)

    for i, b in enumerate(boxes):
        # draw bounding box
        x1 = int(b[0])
        y1 = int(b[1])
        x2 = int(b[2]) + x1
        y2 = int(b[3]) + y1

        color = colors[int(labels[i])%len(colors)]
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)

        # draw label
        label = label_dict[int(labels[i])]
        cv2.putText(img_np, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2_imshow(img_np)
    return img_np
