import numpy as np
import torch, cv2, time
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer


def draw_caption(image, box, caption):
	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def main():
	# Load the csv file and make it to a dataloader
	dataset_val = CSVDataset(train_file='../visDrone_valid.csv', class_list='../classes.csv', transform=transforms.Compose([Normalizer(), Resizer()]))
	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)

	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)


	# Load the model
	retinanet = torch.load('../trained_model/model_final.pt')
	retinanet = retinanet.cuda()
	retinanet.eval()

	unnormalize = UnNormalizer()

	for idx, data in enumerate(dataloader_val):
		with torch.no_grad():
			st = time.time()
			if torch.cuda.is_available():
				scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
			else:
				scores, classification, transformed_anchors = retinanet(data['img'].float())
			print('Elapsed time: {}'.format(time.time() - st))
			idxs = np.where(scores.cpu() > 0.5)

			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

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
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				draw_caption(img, (x1, y1, x2, y2), label_name)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
				print(label_name)

			cv2.imshow('img', img)
			cv2.waitKey(0)

if __name__ == '__main__':
	main()