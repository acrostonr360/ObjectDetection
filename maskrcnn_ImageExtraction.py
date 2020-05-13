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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", default="/home/bmarti/keras-mask-rcnn/mask_rcnn_coco.h5",
	help="path to Mask R-CNN model weights pre-trained on COCO")
ap.add_argument("-l", "--labels", default="/home/bmarti/keras-mask-rcnn/coco_labels.txt",
	help="path to class labels file")
ap.add_argument("-i", "--input", required=True,
	help="path to input folder")
ap.add_argument("-o", "--output", required=True,
    help="path to output video folder")
ap.add_argument("-v", "--videoType", required=True,
	help="the type of video, between normal, flip, stereo and flipStereo")
ap.add_argument("-d", "--detectionMode", required=True,
	help="the method of detection we want to apply, mask, maskLabel, maskHuman, blurHuman or test")
ap.add_argument("-c", "--camera", required=True,
    help="name of the camera")
ap.add_argument("-s", "--start", default=0,
    help="Which frame we start extracting(we are usually not moving during the frist frames)")
ap.add_argument("-e", "--end", default=100000,
    help="Which frame we stop extracting(we are usually not moving during the last frames)")
ap.add_argument("-m", "--modulo", required=True,
    help="the modulo we apply to define the 'jump' we make between frames(we do not extract every frames)")
ap.add_argument("-f", "--fps_ratio", default=5,
    help="the fps ratio between stereo and non-stereo cam)")
ap.add_argument("-t", "--treshold", required=True,
	help="the threshold above wich we accept a detection as valid, between 0..1)")
ap.add_argument("-r", "--record", default="NO",
	help="do we want to record a video?")
args = vars(ap.parse_args())

if not args["detectionMode"]=="mask" and not args["detectionMode"]=="maskLabel" and not args["detectionMode"]=="maskHuman" and not args["detectionMode"]=="blurHuman" and not args["detectionMode"]=="blurMask" and not args["detectionMode"]=="test":
	print("[INFO] detectionMode uncorrect, use: mask, maskLabel, maskHuman, blurHuman or test")
	exit()
if not args["videoType"]=="normal" and not args["videoType"]=="flip" and not args["videoType"]=="stereo" and not args["videoType"]=="flipStereo" and not args["videoType"]=="lightSensorStereo":
	print("[INFO] videoType uncorrect, use: normal, flip, stereo, flipStereo or lightSensorStereo")
	exit()
if not args["record"]=="NO" and not args["record"]=="YES":
	print("[INFO] record uncorrect, use: YES or NO")
	exit()	
	
startTimeGlobal = time.time()
	
	
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

	# number of classes (we would normally add +1 for the background
	# but the background class is *already* included in the class
	# names)
	NUM_CLASSES = len(CLASS_NAMES)

# initialize the inference configuration
config = SimpleConfig()
# initialize the Mask R-CNN model for inference and then load the
# weights
print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config,
	model_dir=os.getcwd())
model.load_weights(args["weights"], by_name=True)


#Read the Input video
vs = cv2.VideoCapture(args["input"])
(W, H) = (None, None)
print("____________________________________________________________________________________________________________")
print("[INFO] try to read file: "+args["input"])
# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO]  {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO]  could not determine # of frames in video")
	print("[INFO]  no approx. completion time can be provided")
	total = -1
	
	
if args["detectionMode"]=="mask":
	output = args["output"]+"/"+args["camera"]+"/Mask_rcnn"
	outputVideo = args["output"]+"/"+args["camera"]+"_Mask_rcnn.avi"
	outputStereoLeft = args["output"]+"/"+args["camera"]+"_Left/Mask_rcnn"
	outputStereoRight = args["output"]+"/"+args["camera"]+"_Right/Mask_rcnn"
elif args["detectionMode"]=="maskLabel":
	output = args["output"]+"/"+args["camera"]+"/MaskLabel_rcnn"
	outputVideo = args["output"]+"/"+args["camera"]+"_MaskLabel_rcnn.avi"
	outputStereoLeft = args["output"]+"/"+args["camera"]+"_Left/MaskLabel_rcnn"
	outputStereoRight = args["output"]+"/"+args["camera"]+"_Right/MaskLabel_rcnn"
elif args["detectionMode"]=="maskHuman":
	output = args["output"]+"/"+args["camera"]+"/MaskHuman_rcnn"
	outputVideo = args["output"]+"/"+args["camera"]+"_MaskHuman_rcnn.avi"
	outputStereoLeft = args["output"]+"/"+args["camera"]+"_Left/MaskHuman_rcnn"
	outputStereoRight = args["output"]+"/"+args["camera"]+"_Right/MaskHuman_rcnn"
elif args["detectionMode"]=="blurHuman":
	output = args["output"]+"/"+args["camera"]+"/BlurMaskHuman_rcnn"
	outputVideo = args["output"]+"/"+args["camera"]+"_BlurMaskHuman_rcnn.avi"
	outputStereoLeft = args["output"]+"/"+args["camera"]+"_Left/BlurMaskHuman_rcnn"
	outputStereoRight = args["output"]+"/"+args["camera"]+"_Right/BlurMaskHuman_rcnn"
elif args["detectionMode"]=="test":
	output = args["output"]+"/"+args["camera"]+"/MaskChair_rcnn"
	outputVideo = args["output"]+"/"+args["camera"]+"_MaskChair_rcnn.avi"
	outputStereoLeft = args["output"]+"/"+args["camera"]+"_Left/MaskChair_rcnn"
	outputStereoRight = args["output"]+"/"+args["camera"]+"_Right/MaskChair_rcnn"
	
videoWriter = None
treshold = float(args["treshold"])
nbFrame = 0
nbFrameExtracted = 0
isLastRight = 1
if args["videoType"]=="stereo" or args["videoType"]=="flipStereo" or args["videoType"]=="lightSensorStereo":	
	startFrame = int(args["start"])*int(args["fps_ratio"])
	endFrame = int(args["end"])*int(args["fps_ratio"])
	modulo = int(args["modulo"])*int(args["fps_ratio"])/2
	if not os.path.isdir(outputStereoLeft) or not os.path.isdir(outputStereoRight):
		os.makedirs(outputStereoLeft)
		os.makedirs(outputStereoRight)
		print("[INFO] Create dir:", outputStereoLeft)
		print("[INFO] Create dir:", outputStereoRight)
else:
	startFrame = int(args["start"])
	endFrame = int(args["end"])
	modulo = int(args["modulo"])
	if not os.path.isdir(output):
		os.makedirs(output)	
		print("[INFO] Create dir:", output)
		
# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		print("[INFO] Not able to grab the frame:", nbFrame)
		break        
	if nbFrame>=endFrame: 
		print("[INFO] We reach the last frame we wanted:", endFrame)
		break		
	if nbFrame<startFrame or not int(nbFrame%modulo)==0:
		nbFrame+=1
		continue	
		
	startTime = time.time()
	
	#First debair images	
	H = int(frame.shape[0])
	W = int(frame.shape[1])
	if args["videoType"]=="normal":
		b,g,r = cv2.split(frame)
		frame = cv2.cvtColor(b, cv2.COLOR_BayerRG2RGB)
	elif args["videoType"]=="flip":
		b,g,r = cv2.split(frame)
		frame = cv2.cvtColor(b, cv2.COLOR_BayerBG2RGB)
	elif args["videoType"]=="stereo":	
		leftFrame = frame[0:H, 0:int(W/2)]
		rightFrame = frame[0:H, int(W/2):W]
		if isLastRight:
			b,g,r = cv2.split(leftFrame)
			frame = cv2.cvtColor(b, cv2.COLOR_BayerRG2RGB)
			leftFrame=frame
			
			b,g,r = cv2.split(rightFrame)
			rightFrame = cv2.cvtColor(b, cv2.COLOR_BayerRG2RGB)
		else:
			b,g,r = cv2.split(rightFrame)
			frame = cv2.cvtColor(b, cv2.COLOR_BayerRG2RGB)
			rightFrame = frame
			
			b,g,r = cv2.split(leftFrame)
			leftFrame = cv2.cvtColor(b, cv2.COLOR_BayerRG2RGB)
	elif args["videoType"]=="flipStereo":
		leftFrame = frame[0:H, 0:int(W/2)]
		rightFrame = frame[0:H, int(W/2):W]
		if isLastRight:
			b,g,r = cv2.split(leftFrame)
			frame = cv2.cvtColor(b, cv2.COLOR_BayerRG2RGB)
			leftFrame=frame
			
			b,g,r = cv2.split(rightFrame)
			rightFrame = cv2.cvtColor(b, cv2.COLOR_BayerBG2RGB)
		else:
			b,g,r = cv2.split(rightFrame)
			frame = cv2.cvtColor(b, cv2.COLOR_BayerBG2RGB)
			rightFrame = frame
			
			b,g,r = cv2.split(leftFrame)
			leftFrame = cv2.cvtColor(b, cv2.COLOR_BayerRG2RGB)
	elif args["videoType"]=="lightSensorStereo":
		leftFrame = frame[0:H, 0:int(W/2)]
		rightFrame = frame[0:H, int(W/2):W]
		if isLastRight:
			b,g,r = cv2.split(leftFrame)
			frame = cv2.cvtColor(b, cv2.COLOR_BayerBG2RGB)
			leftFrame=frame
			
			b,g,r = cv2.split(rightFrame)
			rightFrame = cv2.cvtColor(b, cv2.COLOR_BayerRG2RGB)
		else:
			b,g,r = cv2.split(rightFrame)
			frame = cv2.cvtColor(b, cv2.COLOR_BayerRG2RGB)
			rightFrame = frame
			
			b,g,r = cv2.split(leftFrame)
			leftFrame = cv2.cvtColor(b, cv2.COLOR_BayerBG2RGB)
	if args["record"]=="YES":
		if args["videoType"]=="stereo" or args["videoType"]=="flipStereo" or args["videoType"]=="lightSensorStereo":
				frame = np.concatenate((leftFrame, rightFrame), axis=1)
		if videoWriter is None: 
			fourcc = cv2.VideoWriter_fourcc(*"FFV1")
			videoWriter = cv2.VideoWriter(outputVideo, fourcc, 1, (frame.shape[1], frame.shape[0]), True)
			print("[INFO]  Start to record video:",outputVideo)
		
			
	W = int(frame.shape[1])			
	frameResize = imutils.resize(frame, int(W/2))	
	completeMaskResize=np.zeros([frameResize.shape[0],frameResize.shape[1]],dtype="uint8")
	
	if args["detectionMode"]=="blurHuman":
		frameBlured = cv2.blur(frame,(30,30))
			
	# perform a forward pass of the network to obtain the results
	print("[INFO]  making predictions with Mask R-CNN, frame:",nbFrame)
	r = model.detect([frameResize], verbose=1)[0]
	
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
				if classID==1 or classID==57 or classID==59 or classID==61:
					frameResize = visualize.apply_mask(frameResize, mask, color, alpha=0.3)
			elif args["detectionMode"]=="maskLabel":
				frameResize = visualize.apply_mask(frameResize, mask, color, alpha=0.6)
				(startY, startX, endY, endX) = r["rois"][i]
				label = CLASS_NAMES[classID]
				text = "{}: {:.3f}".format(label, score)
				cv2.rectangle(frameResize, (startX, startY), (endX, endY), color, 2)			
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.putText(frameResize, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
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
	if args["detectionMode"]=="blurHuman":
		completeMask = imutils.resize(completeMaskResize, W)		
		frameBlurMasked = cv2.bitwise_and(frameBlured, frameBlured, mask = completeMask)	
		frameBlurMasked = visualize.apply_mask(frameBlurMasked, completeMask, [1,0,1], alpha=0.05)	
		completeMaskInv = completeMask
		i=0
		y=0
		for i in range (0, H):
			for y in range (0, W):
				if completeMaskInv[i,y] > 0:
					completeMaskInv[i,y] = 0
				else:
					completeMaskInv[i,y] = 255
		frameMasked = cv2.bitwise_and(frame, frame, mask = completeMaskInv)
		frame = cv2.add(frameMasked, frameBlurMasked)		
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	elif args["detectionMode"]=="maskHuman":
		completeMask = imutils.resize(completeMaskResize, W)
		frame = visualize.apply_mask(frame, completeMask, [0,0,0], alpha=1)		
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	elif args["detectionMode"]=="test":
		completeMask = imutils.resize(completeMaskResize, W)	
		#frame = cv2.bitwise_and(frame, frame, mask = completeMask)#To mask the rest of the frame
		#frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
		# Then assign the mask to the last channel of the image
		#frame[:, :, 3] = completeMask
		frame = visualize.apply_mask(frame, completeMask, [0,0,0], alpha=1)		
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	else:
		frame = imutils.resize(frameResize, W)
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	
	if args["record"]=="YES":
		videoWriter.write(frame)
		if isLastRight:
			frame = frame[0:H, 0:int(W/2)]
		else:
			frame = frame[0:H, int(W/2):W]
			
	if args["videoType"]=="stereo" or args["videoType"]=="flipStereo" or args["videoType"]=="lightSensorStereo":	
		if isLastRight:
			print("[INFO] Save frame:", outputStereoLeft+"/"+args["camera"]+"_Left_"+format(nbFrame, '0>5')+".png")
			cv2.imwrite(outputStereoLeft+"/"+args["camera"]+"_Left_"+format(nbFrame, '0>5')+".png", frame)
			isLastRight=0
		else:
			print("[INFO] Save frame:", outputStereoRight+"/"+args["camera"]+"_Right_"+format(nbFrame, '0>5')+".png")
			cv2.imwrite(outputStereoRight+"/"+args["camera"]+"_Right_"+format(nbFrame, '0>5')+".png", frame)
			isLastRight=1
	else:
		print("[INFO] Save frame:", output+"/"+args["camera"]+"_"+format(nbFrame, '0>5')+".png")
		cv2.imwrite(output+"/"+args["camera"]+"_"+format(nbFrame, '0>5')+".png", frame)				
					
	endTime = time.time()
	# some information on processing single frame
	if nbFrame==startFrame:
		if total > 0:
			elap = (endTime - startTime)
			print("[INFO]  single frame took {:.4f} seconds".format(elap))
			print("[INFO]  estimated total time to finish: {:.4f}".format(
			elap * total/modulo))
			
	nbFrameExtracted+=1
	nbFrame+=1

endTimeGlobal = time.time()
print("[INFO] \n_____________________________Process End_________________________________")
print("[INFO] We extracted "+str(nbFrameExtracted)+" frames")
print("[INFO] Over a total of "+str(nbFrame)+" frames")
elap = (endTimeGlobal - startTimeGlobal)
print("[INFO] In {:.4f} seconds".format(elap))
vs.release()
videoWriter.release()