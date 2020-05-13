# import the necessary packages
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", default="/home/bmarti/keras-mask-rcnn/mask_rcnn_coco.h5",
	help="path to Mask R-CNN model weights pre-trained on COCO")
ap.add_argument("-l", "--labels", default="/home/bmarti/keras-mask-rcnn/coco_labels.txt",
	help="path to class labels file")
ap.add_argument("-i", "--input", required=True,
	help="path to input folder")
ap.add_argument("-o", "--output", default=".",
    help="path to output video folder")
ap.add_argument("-d", "--detectionMode", required=True,
	help="the method of detection we want to apply, mask, maskLabel, maskHuman, blurHuman or test")
ap.add_argument("-c", "--camera", required=True,
    help="name of the camera")
ap.add_argument("-t", "--treshold", required=True,
	help="the threshold above wich we accept a detection as valid, between 0..1)")
ap.add_argument("-v", "--video", default="NO",
	help="do we want to export a video?")
ap.add_argument("-r", "--resolution", default="2",
	help="What factor of resolution do we want? 1=normal size, 2=half size, etc...")
args = vars(ap.parse_args())

if not args["detectionMode"]=="mask" and not args["detectionMode"]=="maskLabel" and not args["detectionMode"]=="maskHuman" and not args["detectionMode"]=="blurHuman" and not args["detectionMode"]=="blurMask" and not args["detectionMode"]=="test":
	print("[INFO] detectionMode uncorrect, use: mask, maskLabel, maskHuman, blurHuman or test")
	exit()
if not args["video"]=="NO" and not args["video"]=="YES":
	print("[INFO] video arg uncorrect, use: YES or NO")
	exit()	
	
startTimeGlobal = time.time()
	
if args["output"]==".":
	output = args["input"]
else:
	output = args["output"]
#Initialize mask_rcnn 
# load the class label names from disk, one label per line
CLASS_NAMES = open(args["labels"]).read().strip().split("\n")
# generate random (but visually distinct) colors for each class label
# (thanks to Matterport Mask R-CNN for the method!)
hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
random.seed(42)
random.shuffle(COLORS)

class SimpleConfig(Config):
	# give the configuration a recognizable name
	NAME = "coco_inference"

	# set the number of GPUs to use along with the number of images
	# per GPU	
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	
	allow_growth = True

	# number of classes (we would normally add +1 for the background
	# but the background class is *already* included in the class
	# names)
	NUM_CLASSES = len(CLASS_NAMES)

# initialize the inference configuration
config = SimpleConfig()

#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

# initialize the Mask R-CNN model for inference and then load the
# weights
print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config,
	model_dir=os.getcwd())
model.load_weights(args["weights"], by_name=True)

	
if args["detectionMode"]=="mask":
	output = output+"/"+args["camera"]+"/Mask_rcnn"
	outputVideo = output+"/"+args["camera"]+"_Mask_rcnn.avi"
elif args["detectionMode"]=="maskLabel":
	output = output+"/"+args["camera"]+"/MaskLabel_rcnn"
	outputVideo = output+"/"+args["camera"]+"_MaskLabel_rcnn.avi"
elif args["detectionMode"]=="maskHuman":
	output = output+"/"+args["camera"]+"/MaskHuman_rcnn"
	outputVideo = output+"/"+args["camera"]+"_MaskHuman_rcnn.avi"
elif args["detectionMode"]=="blurHuman":
	output = output+"/"+args["camera"]+"/BlurMaskHuman_rcnn"
	outputVideo = output+"/"+args["camera"]+"_BlurMaskHuman_rcnn.avi"
elif args["detectionMode"]=="test":
	output = output+"/"+args["camera"]+"/MaskChair_rcnn"
	outputVideo = output+"/"+args["camera"]+"_MaskChair_rcnn.avi"
	
videoWriter = None
treshold = float(args["treshold"])
totNbImg = 0

countHuman = 0

if not os.path.isdir(output):
	os.makedirs(output)	
	print("[INFO] Create dir:", output)

Files = os.listdir(args["input"]+"/"+args["camera"])
for name in Files:		
	split = name.split('.')
	if not split[-1] == 'png':
		continue
		
	print("[INFO] File: "+args["input"]+"/"+args["camera"]+"/"+name)
		
	image = cv2.imread(args["input"]+"/"+args["camera"]+"/"+name, cv2.IMREAD_UNCHANGED)
	totNbImg+=1
		
	startTime = time.time()
	
	nbImage = split[0].split('_')[-1]
	#First debair images	
	H = int(image.shape[0])
	W = int(image.shape[1])
	
	if args["video"]=="YES":
		if videoWriter is None: 
			fourcc = cv2.VideoWriter_fourcc(*"FFV1")
			videoWriter = cv2.VideoWriter(outputVideo, fourcc, 1, (image.shape[1], image.shape[0]), True)
			print("[INFO]  Start to record video:",outputVideo)
		
			
	W = int(image.shape[1])			
	imageResize = imutils.resize(image, int(W/float(args["resolution"])))	
	imageResize = cv2.cvtColor(imageResize, cv2.COLOR_BGR2RGB)
	completeMaskResize=np.zeros([imageResize.shape[0],imageResize.shape[1]],dtype="uint8")
	
	if args["detectionMode"]=="blurHuman":
		imageBlured = cv2.blur(image,(30,30))
			
	# perform a forward pass of the network to obtain the results
	print("[INFO]  making predictions with Mask R-CNN, frame:",nbImage)
	r = model.detect([imageResize], verbose=1)[0]
	
	# loop over of the detected object's bounding boxes and masks
	for i in range(0, r["rois"].shape[0]):
		# extract the class ID and mask for the current detection, then
		# grab the color to visualize the mask (in BGR format)
		classID = r["class_ids"][i]
		mask = r["masks"][:, :, i]
		color = COLORS[classID][::-1]
		score = r["scores"][i]
		# visualize the pixel-wise mask of the object
		if score >= treshold:
			if args["detectionMode"]=="mask":
				if classID==1:
					countHuman+=1
					imageResize = visualize.apply_mask(imageResize, mask, [1,0,1], alpha=0.15)
				if classID==57:
					imageResize = visualize.apply_mask(imageResize, mask, [0,0,0.8], alpha=0.3)
				if classID==59:
					imageResize = visualize.apply_mask(imageResize, mask, [0.2,1,0.2], alpha=0.5)
				if classID==61:
					imageResize = visualize.apply_mask(imageResize, mask, [0.6,0.6,0.2], alpha=0.4)
			elif args["detectionMode"]=="maskLabel":
				imageResize = visualize.apply_mask(imageResize, mask, color, alpha=0.6)
				(startY, startX, endY, endX) = r["rois"][i]
				label = CLASS_NAMES[classID]
				text = "{}: {:.3f}".format(label, score)
				cv2.rectangle(imageResize, (startX, startY), (endX, endY), color, 2)			
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.putText(imageResize, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
					0.6, color, 2)
			elif args["detectionMode"]=="test":
				if classID==57:
					mask = np.uint8(mask)
				#	(startY, startX, endY, endX) = r["rois"][i]
				#	cv2.rectangle(mask, (startX, startY), (endX, endY), 255, -1)
					completeMaskResize = cv2.add(completeMaskResize, mask)
			elif classID==1:
				mask = np.uint8(mask)
				completeMaskResize = cv2.add(completeMaskResize, mask)
	
	# convert the image back to BGR so we can use OpenCV's drawing
	# functions
	imageResize = cv2.cvtColor(imageResize, cv2.COLOR_RGB2BGR)
	if args["detectionMode"]=="blurHuman":
		completeMask = imutils.resize(completeMaskResize, W)		
		imageBlurMasked = cv2.bitwise_and(imageBlured, imageBlured, mask = completeMask)	
		imageBlurMasked = visualize.apply_mask(imageBlurMasked, completeMask, [1,0,1], alpha=0.05)	
		completeMaskInv = completeMask
		i=0
		y=0
		for i in range (0, H):
			for y in range (0, W):
				if completeMaskInv[i,y] > 0:
					completeMaskInv[i,y] = 0
				else:
					completeMaskInv[i,y] = 255
		imageMasked = cv2.bitwise_and(image, image, mask = completeMaskInv)
		image = cv2.add(imageMasked, imageBlurMasked)	
	elif args["detectionMode"]=="maskHuman":
		completeMask = imutils.resize(completeMaskResize, W)
		image = visualize.apply_mask(image, completeMask, [0,0,0], alpha=1)	
	elif args["detectionMode"]=="test":
		completeMask = imutils.resize(completeMaskResize, W)	
		#image = cv2.bitwise_and(image, image, mask = completeMask)#To mask the rest of the image
		#image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)
		# Then assign the mask to the last channel of the image
		#image[:, :, 3] = completeMask
		image = visualize.apply_mask(image, completeMask, [0,0,0], alpha=1)		
	else:
		image = imutils.resize(imageResize, W)
	
	if args["video"]=="YES":
		videoWriter.write(image)
			
	print("[INFO] Save frame:", output+"/"+args["camera"]+"_"+format(nbImage, '0>5')+".png")
	cv2.imwrite(output+"/"+args["camera"]+"_"+format(nbImage, '0>5')+".png", image)				
					
	endTime = time.time()
	# some information on processing single frame
	if nbImage == 1:
		elap = (endTime - startTime)
		print("[INFO]  single frame took {:.4f} seconds".format(elap))
		print("[INFO]  estimated total time to finish: {:.4f}".format(
		elap * Files.size()/modulo))

endTimeGlobal = time.time()
print("[INFO] \n_____________________________Process End_________________________________")

print("[INFO] We detected "+str(countHuman)+" people during the scan. Please take into acount that some of those maybe be duplicate or false-positiv")

print("[INFO] Segmented "+str(totNbImg)+" images")
elap = (endTimeGlobal - startTimeGlobal)
print("[INFO] In {:.4f} seconds".format(elap))
if args["video"]=="YES":
	videoWriter.release()