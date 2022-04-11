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


if __name__ == '__main__':

	labels = {0: 'ignored-regions',
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
	retinanet.training = False
	retinanet.eval()

	# A list to store the inference images
	list_inference = []

	# Process the video
	cap = cv2.VideoCapture('../video.mp4')

	start_time = time.time()

	num_frame = 0
	while (cap.isOpened()):
		ret, frame = cap.read()
		if ret == False:
			break
		frame_orig = frame.copy()

		num_frame += 1

		# rescale the frame so the smallest side is min_side
		rows, cols, cns = frame.shape
		smallest_side = min(rows, cols)
		min_side = 608
		max_side = 1024
		scale = min_side / smallest_side

		# check if the largest side is now greater than max_side, which can happen
		# when images have a large aspect ratio
		largest_side = max(rows, cols)

		if largest_side * scale > max_side:
			scale = max_side / largest_side

		# resize the image with the computed scale
		frame = cv2.resize(frame, (int(round(cols * scale)), int(round((rows * scale)))))
		rows, cols, cns = frame.shape

		pad_w = 32 - rows % 32
		pad_h = 32 - cols % 32

		new_frame = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
		new_frame[:rows, :cols, :] = frame.astype(np.float32)
		frame = new_frame.astype(np.float32)
		frame /= 255
		frame -= [0.485, 0.456, 0.406]
		frame /= [0.229, 0.224, 0.225]
		frame = np.expand_dims(frame, 0)
		frame = np.transpose(frame, (0, 3, 1, 2))

		with torch.no_grad():

			frame = torch.from_numpy(frame)
			if torch.cuda.is_available():
				frame = frame.cuda()

			st = time.time()
			scores, classification, transformed_anchors = retinanet(frame.cuda().float())
			print('Elapsed time for this frame: {:.4f} seconds'.format(time.time() - st))
			idxs = np.where(scores.cpu() > 0.5)

			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]

				x1 = int(bbox[0] / scale)
				y1 = int(bbox[1] / scale)
				x2 = int(bbox[2] / scale)
				y2 = int(bbox[3] / scale)
				label_name = labels[int(classification[idxs[0][j]])]
				# print(bbox, classification.shape)
				score = scores[j]
				caption = '{} {:.3f}'.format(label_name, score)
				# draw_caption(img, (x1, y1, x2, y2), label_name)
				draw_caption(frame_orig, (x1, y1, x2, y2), caption)
				cv2.rectangle(frame_orig, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

			# append result to the list
			height, width, layers = frame_orig.shape
			size = (width, height)
			list_inference.append(frame_orig)

			cv2.waitKey(0)

	time_cost = time.time() - start_time
	print('The video gets {} frames.'.format(num_frame))
	print('The total time to process the video is: {:.4f} seconds.'.format(time_cost))
	print('Average time cost for each frame: {:.4f} seconds.'.format(time_cost / num_frame))

	cap.release()

	out = cv2.VideoWriter('prediction.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
	for i in range(len(list_inference)):
		out.write(list_inference[i])
	out.release()
	cv2.destroyAllWindows()
