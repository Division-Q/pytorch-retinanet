import torch
import time
import copy
import cv2, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms

# Define labels
label_dict = {0: 'ignored-regions',
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

# Define data_transform
data_transform = {'train': transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ]),
                  'validation': transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ]),
                  'holdout': transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                      ])
                 }

def reshape_whc2cwh(img):
    w, h, c = img.shape
    img = img.reshape(c, w, h)
    return img

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df=None, transform=None, dir_dataset=None):
        """
        Args:
            df (dataframe): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            dir_dataset (string): The dataset root directory.
        """
        self.df = df
        self.names = np.unique(self.df.iloc[:,0])
        self.df_len = len(self.names)
        self.dir_dataset = dir_dataset
        self.transform = transform

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # Read the image and the label by the index
        path_img = self.names[index]
        dict_label = {}
        df_img = self.df.loc[self.df['img_path'] == path_img]
        boxes = torch.tensor(
            [df_img['x1'].tolist(), df_img['y1'].tolist(), df_img['x2'].tolist(), df_img['y2'].tolist()]).T
        dict_label['boxes'] = boxes.cuda()
        dict_label['labels'] = torch.tensor(df_img['label_index'].tolist()).cuda()
        path_img = os.path.join(self.dir_dataset, path_img)
        img = cv2.imread(path_img)
        img = np.float32(img)

        # Do the image augmentation
        if self.transform:
            img = self.transform(img)
            img = img.cuda()

        sample = {'image': img, 'label': dict_label, 'path_img': path_img}
        return sample

    def __len__(self):
        return self.df_len

# Set up collate function for dataloader
def collate_wrapper(batch):
    imgs = []
    labels = []
    paths_img = []
    for sample in batch:
        imgs.append(sample['image'])
        labels.append(sample['label'])
        paths_img.append(sample['path_img'])
    return {'images': imgs, 'labels': labels, 'path_img': paths_img}

def annotate_img(img_np, pred_dict):
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (255,255,255)]

    # add annotations
    boxes = pred_dict['boxes'].cpu()
    labels = pred_dict['labels'].cpu()
    #print(labels)

    for i, b in enumerate(boxes):
        # draw bounding box
        x1 = int(b[0])
        y1 = int(b[1])
        x2 = int(b[2])
        y2 = int(b[3])

        color = colors[int(labels[i])%len(colors)]
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)

        # draw label
        label = label_dict[int(labels[i])]
        cv2.putText(img_np, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    cv2_imshow(img_np)
    return img_np

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


def dataset2df(dir_dataset, img_paths, ann_paths):
    '''
    :param dir_dataset: string, path to parent dataset dir
    :param img_paths: list of strings
    :param ann_paths: list of strings
    :return: dataframe with each image and annotation
    '''
    data = []
    for img_path, ann_path in zip(img_paths, ann_paths):
        with open(os.path.join(dir_dataset,ann_path), 'r') as f:
            lines = f.readlines()

            for line in lines:
                raw = line.split(',')
                if int(raw[4]) == 1:
                    x1 = int(raw[0])
                    y1 = int(raw[1])
                    x2 = int(raw[2]) + x1
                    y2 = int(raw[3]) + y1
                    label_index = int(raw[5])
                    label = label_dict[label_index]
                    data.append([img_path, x1, y1, x2, y2, label_index, label])

    return pd.DataFrame(data, columns=['img_path', 'x1', 'y1', 'x2', 'y2', 'label_index', 'label'])

def get_names(dir_dataset, n):
    '''
    get n random image names in the dataset
    :param dir_dataset: string path to dataset
    :param n: int
    :return: list of filenames (no extension)
    '''
    dir_imgs = os.path.join(dir_dataset, 'images')
    all_images = os.listdir(dir_imgs)
    total_images = len(all_images)

    # pick subset random integers in range
    sample = np.random.choice(range(total_images), n, replace=False)

    sample_images = [all_images[s].split('.')[0] for s in sample]
    return sample_images

def IoU(bb1, bb2):
    iou = 0
    return iou

def train(model, df_train, df_valid, dir_dataset, epochs, batch_size, loss_func, optimizer):
    # Put the model in the GPU
    model = model.cuda()
    epochs_training_loss = np.array([], dtype='float64')
    epochs_val_loss = np.array([], dtype='float64')
    model_best = model
    loss_best = 100000
    epoch_best = 0
    train_accuracy = []
    val_accuracy = []
    history = {}

    # Set up dataset instances
    train_set = Dataset(df=df_train, transform=data_transform['train'], dir_dataset=dir_dataset)
    valid_set = Dataset(df=df_valid, transform=data_transform['validation'], dir_dataset=dir_dataset)

    # Set up dataloader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, collate_fn=collate_wrapper)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, collate_fn=collate_wrapper)

    samples = next(iter(train_loader))
    print(samples)
    imgs = samples['images']
    labels = samples['labels']
    #paths_img = samples['path_img']

    for e in range(epochs):

        torch.cuda.empty_cache()
        # Calculate the time
        epoch_start = time.time()

        print("In epoch", e)
        train_running_loss = 0
        val_running_loss = 0

        right_predict = 0

        for batch in iter(train_loader):  # processing one batch at a time

            images = batch['images']
            labels = batch['labels']
            paths_img = batch['path_img']

            # correct the slashes in the path
            paths_img = [os.path.abspath(p) for p in paths_img]

            # get the loss for the prediction vs ground truth
            try:
                loss_dict = model(images, labels)
            except Exception as e:
                print('Exception: ')
                print(e)
                print('Data {} is bad'.format(paths_img))
                continue
            # TODO: update loss for each subnet individually
            loss = sum(loss for loss in loss_dict.values())
            #print(loss)

            # TODO: right prediction if IoU of BB > 0.5
            #right_predict += IoU()

            # BACK PROPAGATION OF LOSS to generate updated weights
            # Zero the gradients before training
            optimizer.zero_grad()
            # Calculate the gradients of loss
            loss.backward()
            # update the weight
            optimizer.step()

            train_running_loss += loss.item()
        #train_accuracy.append((right_predict / df_train.shape[0]).cpu().numpy())

        # TODO: evaluate the model
        # Evaluate the model
        # Compute the evaluation loss
        right_predict = 0
        for batch in valid_loader:
            torch.cuda.empty_cache()

            images = batch['images']
            labels = batch['labels']
            # get the loss for the prediction vs ground truth
            loss_dict = model(images, labels)
            # TODO: update loss for each subnet individually
            loss = sum(loss for loss in loss_dict.values())

            val_running_loss += loss.item()

        #val_accuracy.append((right_predict / df_valset.shape[0]).cpu().numpy())

        # set the best epoch
        if val_running_loss / len(valid_loader) < loss_best:
            loss_best = val_running_loss / len(valid_loader)
            model_best = copy.deepcopy(model)
            epoch_best = e

        epochs_training_loss = np.append(epochs_training_loss, train_running_loss / len(train_loader))
        epochs_val_loss = np.append(epochs_val_loss, val_running_loss / len(valid_loader))

        epoch_end = time.time()
        print('\tTime consumption: {:.4f}'.format(epoch_end - epoch_start))
        print('\tTraining loss: {:.4f}'.format(train_running_loss / len(train_loader)))
        print('\tValidation loss: {:.4f}'.format(val_running_loss / len(valid_loader)))

    history['training_loss'] = epochs_training_loss
    history['validation_loss'] = epochs_val_loss
    history['epoch_best'] = epoch_best
    history['loss_best'] = loss_best
    #history['train_accuracy'] = train_accuracy
    #history['val_accuracy'] = val_accuracy

    return(model_best, history)

def draw_loss(history):
    plt.figure(figsize=(10, 7), facecolor='w')
    epochs_training_loss = history['training_loss']
    epochs_val_loss = history['validation_loss']
    epochs = len(epochs_training_loss)
    plt.plot(np.array(range(epochs)) + 1., epochs_training_loss, 'b', label='train loss', lw=2)
    plt.plot(np.array(range(epochs)) + 1., epochs_val_loss, 'r', label='val loss', lw=2)
    plt.axvline(x=history['epoch_best'], linestyle='--', label='best epoch')
    plt.title('Loss graph')
    plt.legend()
    plt.show()
