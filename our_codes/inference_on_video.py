import numpy as np
import torch, cv2, time, os, skimage
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer

def draw_caption(image, box, caption):
	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)



dict_labels = {0: 'ignored-regions',
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
			   11: 'others'}

# Load the model
retinanet = torch.load('../trained_model/model_final.pt')
retinanet = retinanet.cuda()
retinanet.eval()

unnormalize = UnNormalizer()
transform = transforms.Compose([Normalizer(), Resizer()])

cap = cv2.VideoCapture('../video.mp4')
while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == False:
		break

	# preprocessed the frame:
	frame = frame.astype(np.float32) / 255.0
	sample = {'img': frame, 'annot': frame} # Just for transformation, the `annot` value is not important here
	frame_trans = transform(sample)['img']
	frame_ready = frame_trans.view(-1, frame_trans.shape[2], frame_trans.shape[0], frame_trans.shape[1])

	st = time.time()
	if torch.cuda.is_available():
		scores, classification, transformed_anchors = retinanet(frame_ready.cuda().float())
	else:
		scores, classification,  = retinanet(frame_ready.float())
	print('Elapsed time: {}'.format(time.time() - st))
	idxs = np.where(scores.cpu() > 0.5)


	img = np.array(255 * unnormalize(frame_ready[0, :, :, :])).copy()

	img[img < 0] = 0
	img[img > 255] = 255

	img = np.transpose(img, (1, 2, 0))

	img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

	for j in range(idxs[0].shape[0]):
		bbox = transformed_anchors[idxs[0][j], :]
		x1 = int(bbox[0])
		y1 = int(bbox[1])
		x2 = int(bbox[2])
		y2 = int(bbox[3])
		label_name = dict_labels[int(classification[idxs[0][j]])]
		draw_caption(img, (x1, y1, x2, y2), label_name)

		cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
		print(label_name)

	cv2.imshow('img', img)
	cv2.waitKey(0)




cap.release()
cv2.destroyAllWindows()