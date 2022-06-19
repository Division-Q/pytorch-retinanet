import numpy as np
import torch, cv2, time, os, skimage
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")


if __name__ == '__main__':
    # Load the model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # A list to store the inference images
    list_inference = []

    # Process the video
    cap = cv2.VideoCapture('../video.mp4')

    # Calculate the time
    start_time = time.time()
    processing_time = 0.0

    # Path_frame

    num_frame = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        with torch.no_grad():
            frame = torch.from_numpy(frame)
            if torch.cuda.is_available():
                frame = frame.cuda()
            st = time.time()
            cv2.imwrite(os.path.join(path_frames, 'frame_{:05d}.png'.format(num_frame - 1)), frame_orig)
            results = model(frame.view([-1, frame.shape[2], frame.shape[0], frame.shape[1]]))
            elapsed = time.time() - st
            print('Elapsed time for this frame: {:.4f} seconds'.format(elapsed))
            processing_time += elapsed

            # append result to the list
            height, width, layers = frame.shape
            size = (width, height)
            list_inference.append(frame)

    time_cost = time.time() - start_time
    print('The video gets {} frames.'.format(num_frame))
    print('The total time to process the video is: {:.4f} seconds.'.format(time_cost))
    print('Average time cost for each frame: {:.4f} seconds.'.format(processing_time / num_frame))
    print('Cost to only run inference: {:.4f} seconds.'.format(processing_time))

    cap.release()

    out = cv2.VideoWriter('prediction.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, size)
    for i in range(len(list_inference)):
        out.write(list_inference[i])
    out.release()
    cv2.destroyAllWindows()
