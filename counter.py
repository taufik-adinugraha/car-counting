# counter (each type), tracker with saved targets, traffic jam detection (counting total # vehicles)

# import the necessary packages
from libs.centroidtracker import CentroidTracker
from libs.trackableobject import TrackableObject
from libs import myconfig as config
from libs import detection
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
from imutils.video import FPS
import dlib
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
ap.add_argument("-s", "--skip-frames", type=int, default=30, help="# of skip frames between detections")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov4.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov4.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
fps = int(vs.get(cv2.CAP_PROP_FPS))
# print(fps)

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=10, maxDistance=50)
trackers = []
trackableObjects = {}


# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

# start the frames per second throughput estimator
fps = FPS().start()
Nmobil, Nmotor, Nbus, Ntruk = [0, 0], [0, 0], [0, 0], [0, 0]

while True:
	print(f'frame: {totalFrames}')
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	# if the frame was not grabbed, then we have reached the end of the stream
	if not grabbed: 
		break

	frame = imutils.resize(frame, width=700)
	(H, W) = frame.shape[:2]
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# detection lines
	upperline = 5 * H // 9
	bottomline = 6 * H // 9

	# initialize the current status along with our list of bounding
	# box rectangles returned by either (1) our object detector or
	# (2) the correlation trackers
	status = "waiting"
	rects = []
	bboxes = []
	ccentroids = []

	# check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
	if totalFrames % args["skip_frames"] == 0:
		# set the status and initialize our new set of object trackers
		status = "detecting"
		trackers = []
		targets = []

		vehicles = ['car', 'truck', 'bus', 'motorbike']
		results = detection.detect_object(frame, net, ln, Idxs=[LABELS.index(i) for i in vehicles if LABELS.index(i) is not None])

		for (i, (classID, prob, bbox, centroid)) in enumerate(results):
			bboxes.append(bbox)
			ccentroids.append(centroid)
			targets.append(classID)
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			# construct a dlib rectangle object from the bounding
			# box coordinates and then start the dlib correlation
			# tracker
			tracker = dlib.correlation_tracker()
			rect = dlib.rectangle(startX, startY, endX, endY)
			tracker.start_track(rgb, rect)

			# add the tracker to our list of trackers so we can
			# utilize it during skip frames
			trackers.append(tracker)

	# otherwise, we should utilize our object *trackers* rather than
	# object *detectors* to obtain a higher frame processing throughput
	else:
		# loop over the trackers
		for i, tracker in enumerate(trackers):
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "tracking"

			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))
			bboxes.append((startX, startY, endX, endY))
			ccentroids.append((startX+(endX-startX)//2, startY+(endY-startY)//2))

		
	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects, targets)

	# loop over the tracked objects
	for iobject, (objectID, (centroid, target, rect)) in enumerate(objects.items()):

		# check to see if a trackable object exists for the current object ID
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			# ignore centroids beyond the limit lines
			if ((centroid[0] > W//2 and centroid[1] < bottomline and centroid[1] > upperline) or (centroid[0] < W//2 and centroid[1] > upperline)):
				to = TrackableObject(objectID, centroid, target, rect)
				mtarget = target

		# otherwise, there is a trackable object so we can utilize it to determine direction
		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)
			
			# find the most voted/frequent target for each trackable object found in "detecting" mode
			# if objectID == 124:
			# 	print(mtarget, [t for i, t in enumerate(to.targets) if i % args["skip_frames"] == 0])

			if i % args["skip_frames"] != 0:
				mtarget = np.bincount([t for i, t in enumerate(to.targets) if i % args["skip_frames"] == 0]).argmax()
				if ((centroid[0] > W//2 and centroid[1] < bottomline and centroid[1] > upperline) or (centroid[0] < W//2 and centroid[1] > upperline)):
					to.targets.append(mtarget)
			else:
				if ((centroid[0] > W//2 and centroid[1] < bottomline and centroid[1] > upperline) or (centroid[0] < W//2 and centroid[1] > upperline)):
					to.targets.append(target)
				mtarget = target

			label = LABELS[mtarget]
			to.rects.append(rect)

			# check to see if the object has been counted or not
			if not to.counted:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
				if direction < 0 and centroid[1] < upperline and centroid[0] < W//2:
					if label == 'car':
						Nmobil[0] += 1
					elif label == 'truck':
						Ntruk[0] += 1
					elif label == 'motorbike':
						Nmotor[0] += 1
					elif label == 'bus':
						Nbus[0] += 1
					totalUp += 1
					to.counted = True
					
				# if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object
				elif direction > 0 and centroid[1] > bottomline and centroid[0] > W//2:
					if label == 'car':
						Nmobil[1] += 1
					elif label == 'truck':
						Ntruk[1] += 1
					elif label == 'motorbike':
						Nmotor[1] += 1
					elif label == 'bus':
						Nbus[1] += 1
					totalDown += 1
					to.counted = True

		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of theobject on the output frame
		if (centroid[1] < bottomline and centroid[0] > W//2) or (centroid[1] > upperline and centroid[0] < W//2):
			# text = "ID {}".format((objectID, LABELS[mtarget][0]))
			# cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
			# cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

			# draw a bounding box
			(startX, startY, endX, endY) = rect
			label = LABELS[mtarget]
			if label == 'car':
				color = (0, 0, 255)
				text = None
			elif label == 'truck':
				color = (255, 0, 0)
				text = 'truk'
			elif label == 'motorbike':
				color = (255,255,0)
				text = 'motor'
			elif label == 'bus':
				color = (0, 255, 0)
				text = 'bus'
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			if label != 'car':
				# get the width and height of the text box
				font_scale = 0.6
				font = cv2.FONT_HERSHEY_PLAIN
				rectangle_bgr = color
				(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
				# set the text start position
				# y = startY - 10 if startY - 10 > 10 else startY + 10
				text_offset_x, text_offset_y = startX, startY
				# make the coords of the box with a small padding of two pixels
				box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
				cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
				cv2.putText(frame, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(255,255,255), thickness=1)
				# cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)


	# draw lines:
	cv2.line(frame, (0, upperline), (W//2, upperline), (0, 255, 255), 1)
	cv2.line(frame, (W//2, bottomline), (W, bottomline), (0, 255, 255), 1)
	cv2.line(frame, (W//2, bottomline), (W//2, upperline), (0, 255, 255), 1)
	# cv2.line(frame, perspective_line[0], perspective_line[1], (0, 255, 255), 1)

	# count total #vehicles on each side
	Nleft = len([i[0] for i in ccentroids if i[0] < W//2 ])
	Nright = len([i[0] for i in ccentroids if i[0] > W//2])
	status = [0, 0]
	for ii, N in enumerate([Nleft, Nright]):
		if N <= 11:
			status[ii] = "LANCAR"
		elif N > 11 and N <= 15:
			status[ii] = "PADAT"
		elif N > 15:
			status[ii] = "MACET"

	# construct a tuple of information we will be displaying on the
	# frame
	for i, total in enumerate([totalUp, totalDown]):
		info = [
			("Status", status[i]),
			("Mobil", Nmobil[i]),
			("Motor", Nmotor[i]),
			("Bus", Nbus[i]),
			("Truk", Ntruk[i]),
			("Total", total)
		]
		# loop over the info tuples and draw them on our frame
		font_scale = 0.5
		font = cv2.FONT_HERSHEY_PLAIN
		rectangle_bgr = (144,238,144)
		box_coords = (5+i*555, 105), (i*555+140, 5)
		cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
		for (j, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			if v in ['PADAT', 'MACET']:
				ctext = (0,0,255)
			elif k == 'Total':
				ctext = (255,0,0)
			else:
				ctext = (0, 0, 0)
			cv2.putText(frame, text, ((i * 555) + 10, int((.4*j)*H//10)+20),
				cv2.FONT_HERSHEY_SIMPLEX, font_scale, ctext, 1)

	# increment the total number of frames processed thus far and
	# then update the FPS counter
	totalFrames += 1
	fps.update()

	# check to see if the output frame should be displayed to our
	# screen
	if args["display"] > 0:
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# if an output video file path has been supplied and the video
	# writer has not been initialized, do so now
	if args["output"] != "" and writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 25,
			(frame.shape[1], frame.shape[0]), True)

	# if the video writer is not None, write the frame to the output
	# video file
	if writer is not None:
		writer.write(frame)



