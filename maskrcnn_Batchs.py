# import the necessary packages
import argparse
import imutils
import os
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input folder")
ap.add_argument("-o", "--output", default=".",
	help="path to output folder")
ap.add_argument("-d", "--detectionMode", required=True,
	help="the method of detection we want to apply; mask, maskLabel, maskHuman, blurHuman or test")
ap.add_argument("-t", "--treshold", required=True,
	help="the threshold above wich we accept a detection as valid, between 0..1)")
ap.add_argument("-r", "--record", default="NO",
	help="Do we want to record videos?")
args = vars(ap.parse_args())

if not args["detectionMode"]=="mask" and not args["detectionMode"]=="maskLabel" and not args["detectionMode"]=="maskHuman" and not args["detectionMode"]=="blurHuman" and not args["detectionMode"]=="test":
	print("detectionMode uncorrect, use: mask, maskLabel, maskHuman, blurHuman, blurMask or test")
	exit()

print("Input : "+args["input"])
print("Output : "+args["output"])

startTime = time.time()

if args["output"]==".":
	output = args["input"]
else:
	output = args["output"]

#RearStereo
if not os.path.isdir(output+"/RearStereo_Left") and not os.path.isdir(output+"/RearStereo_Right"):
	os.makedirs(output+"/RearStereo_Left")
	os.makedirs(output+"/RearStereo_Right")
os.system("python3 maskrcnn_Batch.py --i "+args["input"]+" --o "+args["output"]+"  --c RearStereo_Left --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
os.system("python3 maskrcnn_Batch.py --i "+args["input"]+" --o "+args["output"]+"  --c RearStereo_Right --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")


#FrontStereo
if not os.path.isdir(output+"/FrontStereo_Left") and not os.path.isdir(output+"/FrontStereo_Right"):
	os.makedirs(output+"/FrontStereo_Left")
	os.makedirs(output+"/FrontStereo_Right")
os.system("python3 maskrcnn_Batch.py --i "+args["input"]+" --o "+args["output"]+"  --c FrontStereo_Left --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
os.system("python3 maskrcnn_Batch.py --i "+args["input"]+" --o "+args["output"]+"  --c FrontStereo_Right --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")

#RearLeft
if not os.path.isdir(output+"/RearLeft"):
	os.makedirs(output+"/RearLeft")
os.system("python3 maskrcnn_Batch.py --i "+args["input"]+" --o "+args["output"]+" --c RearLeft --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")

#RearRight
if not os.path.isdir(output+"/RearRight"):
	os.makedirs(output+"/RearRight")
os.system("python3 maskrcnn_Batch.py --i "+args["input"]+" --o "+args["output"]+"  --c RearRight --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")

#FrontLeft
if not os.path.isdir(output+"/FrontLeft"):
	os.makedirs(output+"/FrontLeft")
os.system("python3 maskrcnn_Batch.py --i "+args["input"]+" --o "+args["output"]+"  --c FrontLeft --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")

#FrontRight
if not os.path.isdir(output+"/FrontRight"):
	os.makedirs(output+"/FrontRight")
os.system("python3 maskrcnn_Batch.py --i "+args["input"]+" --o "+args["output"]+"  --c FrontRight --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")

#TopLeft
if not os.path.isdir(output+"/TopLeft"):
	os.makedirs(output+"/TopLeft")
#os.system("python3 maskrcnn_Batch.py --i "+args["input"]+" --o "+args["output"]+"  --c TopLeft --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")

#TopRight
if not os.path.isdir(output+"/TopRight"):
	os.makedirs(output+"/TopRight")
#os.system("python3 maskrcnn_Batch.py --i "+args["input"]+" --o "+args["output"]+"  --c TopRight --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")

endTime = time.time()
print("Total time for the segmentation and extraction was {:.4f}".format(endTime-startTime))