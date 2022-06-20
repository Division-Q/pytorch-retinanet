# Downloads the requirement python package:
# pip install -qr https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt  # install dependencies

import numpy as np
import pandas as pd
import torch, cv2


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def main():
    df = pd.read_csv('../visDrone_valid.csv')
    paths_img = np.unique(df.iloc[:, 0].tolist())
    # Load the model
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')

    for idx, path_img in enumerate(paths_img):
        results = model(path_img)
        results.show()
        # cv2.imshow('img', results.imgs[0])
        # cv2.waitKey(0)
if __name__ == '__main__':
    main()

