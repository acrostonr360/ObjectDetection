# import the necessary packages
import argparse
import imutils
import os
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input folder")
ap.add_argument("-o", "--output", required=True,
	help="path to output folder")
ap.add_argument("-d", "--detectionMode", required=True,
	help="the method of detection we want to apply; mask, maskLabel, maskHuman, blurHuman or test")
ap.add_argument("-s", "--start", default=0,
	help="the threshold to start extracting frames(we are usually not moving during the frist frames)")
ap.add_argument("-e", "--end", default=100000,
	help="the threshold to ebd extracting frames(we are usually not moving during the last frames)")
ap.add_argument("-m", "--modulo", required=True,
	help="the modulo we apply to define the 'jump' we make between frames(we do not exrtract every frames)")
ap.add_argument("-f", "--fps_ratio", required=True,
	help="the fps ratio between stereo and non-stereo cam)")
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
print("Start : "+args["start"])
#The modulo will be 4x higher for stereo videos because their frame rates is 4x  the others cams(20fps compare to 5fps)
print("Modulo : "+args["modulo"])

startTime = time.time()

#RearStereo
if not os.path.isdir(args["output"]+"/RearStereo_Left") and not os.path.isdir(args["output"]+"/RearStereo_Right"):
	os.makedirs(args["output"]+"/RearStereo_Left")
	os.makedirs(args["output"]+"/RearStereo_Right")
#os.system("python3 maskrcnn_ImageExtraction.py --i "+args["input"]+"/CS_0_1/CS_0_1_chunk_000.avi --o "+args["output"]+"  --c RearStereo --s "+str(args["start"])+" --e "+str(args["end"])+" --m "+str(args["modulo"])+" --f "+str(args["fps_ratio"])+" --v lightSensorStereo --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")


#FrontStereo
if not os.path.isdir(args["output"]+"/FrontStereo_Left") and not os.path.isdir(args["output"]+"/FrontStereo_Right"):
	os.makedirs(args["output"]+"/FrontStereo_Left")
	os.makedirs(args["output"]+"/FrontStereo_Right")
#os.system("python3 maskrcnn_ImageExtraction.py --i "+args["input"]+"/CS_4_5/CS_4_5_chunk_000.avi --o "+args["output"]+"  --c FrontStereo --s "+str(args["start"])+" --e "+str(args["end"])+" --m "+str(args["modulo"])+" --f "+str(args["fps_ratio"])+" --v flipStereo --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")

#RearLeft
if not os.path.isdir(args["output"]+"/RearLeft"):
	os.makedirs(args["output"]+"/RearLeft")
#os.system("python3 maskrcnn_ImageExtraction.py --i "+args["input"]+"/CS_rear_left/CS_rear_left_chunk_000.avi --o "+args["output"]+" --c RearLeft  --s "+str(args["start"])+" --e "+str(args["end"])+" --m "+str(args["modulo"])+" --f "+str(args["fps_ratio"])+" --v normal --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")

#RearRight
if not os.path.isdir(args["output"]+"/RearRight"):
	os.makedirs(args["output"]+"/RearRight")
os.system("python3 maskrcnn_ImageExtraction.py --i "+args["input"]+"/CS_rear_right/CS_rear_right_chunk_000.avi --o "+args["output"]+"  --c RearRight  --s "+str(args["start"])+" --e "+str(args["end"])+" --m "+str(args["modulo"])+" --f "+str(args["fps_ratio"])+" --v flip --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")

#FrontLeft
if not os.path.isdir(args["output"]+"/FrontLeft"):
	os.makedirs(args["output"]+"/FrontLeft")
#os.system("python3 maskrcnn_ImageExtraction.py --i "+args["input"]+"/CS_front_left/CS_front_left_chunk_000.avi --o "+args["output"]+"  --c FrontLeft  --s "+str(args["start"])+" --e "+str(args["end"])+" --m "+str(args["modulo"])+" --f "+str(args["fps_ratio"])+" --v normal --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")

#FrontRight
if not os.path.isdir(args["output"]+"/FrontRight"):
	os.makedirs(args["output"]+"/FrontRight")
#os.system("python3 maskrcnn_ImageExtraction.py --i "+args["input"]+"/CS_front_right/CS_front_right_chunk_000.avi  --o "+args["output"]+"  --c FrontRight  --s "+str(args["start"])+" --e "+str(args["end"])+" --m "+str(args["modulo"])+" --f "+str(args["fps_ratio"])+" --v flip --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")

#TopLeft
if not os.path.isdir(args["output"]+"/TopLeft"):
	os.makedirs(args["output"]+"/TopLeft")
#os.system("python3 maskrcnn_ImageExtraction.py --i "+args["input"]+"/CS_top_left/CS_top_left_chunk_000.avi --o "+args["output"]+"  --c TopLeft  --s "+str(args["start"])+" --e "+str(args["end"])+" --m "+str(args["modulo"])+" --f "+str(args["fps_ratio"])+" --v flip --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")

#TopRight
if not os.path.isdir(args["output"]+"/TopRight"):
	os.makedirs(args["output"]+"/TopRight")
#os.system("python3 maskrcnn_ImageExtraction.py --i "+args["input"]+"/CS_top_right/CS_top_right_chunk_000.avi  --o "+args["output"]+"  --c TopRight  --s "+str(args["start"])+" --e "+str(args["end"])+" --m "+str(args["modulo"])+" --f "+str(args["fps_ratio"])+" --v flip --d "+args["detectionMode"]+" --t "+str(args["treshold"])+" --r "+str(args["record"]))
print("")

endTime = time.time()
print("Total time for the segmentation and extraction was {:.4f}".format(endTime-startTime))