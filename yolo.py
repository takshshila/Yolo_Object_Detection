import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
from yolo_utils import infer_image

FLAGS = []

#python yolo.py -w=yolov3-tiny.weights -cfg=cfg/yolov3-tiny.cfg -v=Marketstreet.mp4 -l=data/coco.names -c=0.4 -t=0.4
#python yolo.py -w=YOLOv3/yolov3-tiny.weights -cfg=YOLOv3/yolov3-tiny.cfg -v=images/Marketstreet.mp4 -l=YOLOv3/coco.names -c=0.4 -t=0.4
#python yolo.py -v=images/Marketstreet.mp4

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-w', '--weights',
		type=str,
		default='yolov3-tiny.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

	parser.add_argument('-cfg', '--config',
		type=str,
		default='cfg/yolov3-tiny.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

	parser.add_argument('-v', '--video-path',
		type=str,
		help='The path to the video file')


	parser.add_argument('-vo', '--video-output-path',
		type=str,
        default='outputs/output.mp4',
		help='The path of the output video file')

	parser.add_argument('-i', '--image-path',
						type=str,
						help='The path of the output image file')


	parser.add_argument('-l', '--labels',
		type=str,
		default='data/coco.names',
		help='Path to the file having the \
					labels in a new-line seperated way.')

	parser.add_argument('-c', '--confidence',
		type=float,
		default=0.5,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

	parser.add_argument('-th', '--threshold',
		type=float,
		default=0.3,
		help='The threshold to use when applying the \
				Non-Max Suppresion')

	FLAGS, unparsed = parser.parse_known_args()

	FLAGS.output="outputs"

	# Get the labels
	labels = open(FLAGS.labels).read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# If both image and video files are given then raise error
	print(f'FLAGS.video_path: {FLAGS.video_path}')

	print(f"FLAGS: {FLAGS}")

	if FLAGS.image_path:

		print(f"FLAGS.image_path : {FLAGS.image_path}")

		image = cv.imread(FLAGS.image_path)
		height, width = image.shape[:2]

		outpath=FLAGS.image_path.split('/')[-1][:-4]

		#print(f"outpath : {outpath}")
		output=FLAGS.output+'/'+outpath+'_YOLOv3_output.jpg'

		outputFile = output+ '_YOLOv3_output.jpg'#FLAGS.image_path[:-4] + '_YOLOv3_output.jpg'
		#print(f"outputFile : {outputFile}")

		frame, _, _, _, _ = infer_image(net, layer_names, height, width, image, colors, labels, FLAGS)
		cv.imwrite(outputFile, frame.astype(np.uint8))



	elif FLAGS.video_path:
		# Read the video
		vid = cv.VideoCapture(str(FLAGS.video_path))
		height, width, writer= None, None, None
		while True:

			grabbed, frame = vid.read()

			if not grabbed:
				break

			if width is None or height is None:
				height, width = frame.shape[:2]

			frame, _, _, _, _ = infer_image(net, layer_names, height, width, frame, colors, labels, FLAGS)

			if writer is None:
				fourcc = cv.VideoWriter_fourcc(*'mp4v')
				writer = cv.VideoWriter(FLAGS.video_output_path, fourcc, 30,(frame.shape[1], frame.shape[0]), True)


			writer.write(frame)


		print ("[INFO] Cleaning up...")
		writer.release()
		vid.release()

	elif FLAGS.video_path is None:
	    print ('Path to video not provided')

	elif FLAGS.image_path is None:
	    print ('Path to image not provided')



	else:
		print("[ERROR] Something's not right...")
