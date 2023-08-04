# based on https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV
# TODO: clean, test, add readme.md, normal commentaries and explain this mess
# USAGE 
# python map_altitude_OCR_tool.py --i input_path


# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import logging
import shutil
import argparse
import time
from datetime import datetime
import pandas as pd
import cv2
import re
import os



def parse_tfw(filename):
    f = open(filename, "r")
    res = {'A':0,'D':0,'B':0,'E':0,'C':0,'F':0}
    res['A'] = float(f.readline())
    res['D'] = float(f.readline())
    res['B'] = float(f.readline())
    res['E'] = float(f.readline())
    res['C'] = float(f.readline())
    res['F'] = float(f.readline())
    return res

def convert_coords(x,y,tfw_dict):
    x_conv = tfw_dict['A'] * x + tfw_dict['B'] * y + tfw_dict['C']
    y_conv = tfw_dict['D'] * x + tfw_dict['E'] * y + tfw_dict['F']
    return x_conv, y_conv

def decode_predictions(scores, geometry, min_confidence):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < min_confidence:
				continue
			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

def check_text(text):
    res = re.findall('\d{2,3}[.]\d{1}',text)
    if len(res) == 0:
        return False
    elif float(res[-1])<50 or float(res[-1])>200:
        return False
    else:
        return res[-1]

def detection_and_recognition(image_path, width, height, east_path, min_conf, padding):
    # load the input image and grab the image dimensions
    image = cv2.imread(image_path)
    orig = image.copy()
    (origH, origW) = image.shape[:2]
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (width, height)
    rW = origW / float(newW)
    rH = origH / float(newH)
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested in -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
	    "feature_fusion/Conv_7/Sigmoid",
	    "feature_fusion/concat_3"]
    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(east_path)

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	    (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    (rects, confidences) = decode_predictions(scores, geometry, min_conf)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    # initialize the list of results
    results = []
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
	    # scale the bounding box coordinates based on the respective
	    # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
	    # in order to obtain a better OCR of the text we can potentially
	    # apply a bit of padding surrounding the bounding box -- here we
	    # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)
	    # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
	    # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]
        # in order to apply Tesseract v4 to OCR text we must supply
	    # (1) a language, (2) an OEM flag of 4, indicating that the we
	    # wish to use the LSTM neural net model for OCR, and finally
	    # (3) an OEM value, in this case, 7 which implies that we are
	    # treating the ROI as a single line of text
        config = ("-l eng --oem 1 --psm 6")
        options = "outputbase digits"
        text = pytesseract.image_to_string(roi, config=options)#config)
        #text = replace_chars(text)
	    # add the bounding box coordinates and OCR'd text to the list
	    # of results
        results.append(((startX, startY, endX, endY), text))
    return results

def process_file_by_name(filename, east_path, width, height, padding, min_conf):
    res_df = pd.DataFrame({'x':[],'y':[],'z':[]})
    tif_filename = filename+'.tif'
    tfw_filename = filename+'.tfw'
    results = detection_and_recognition(tif_filename, width, height, east_path, min_conf, padding)
    results = sorted(results, key=lambda r:r[0][1])
    for ((startX, startY, endX, endY), text) in results:
    # display the text OCR'd by Tesseract
    #print("OCR TEXT")
    #print("========")
    #print("{}\n".format(text))
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV, then draw the text and a bounding box surrounding
	# the text region of the input image
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        text = check_text(text)
        if text:
            x = startX
            y = (startY + endY)//2
            x, y = convert_coords(x,y, parse_tfw(tfw_filename))
            z = float(text)
            res_df = res_df.append({'x':x,'y':y,'z':z},ignore_index=True)
    
    return res_df

# do all stuff
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input_path',        default=None,                                   type=str,   help='Path to a image, directory containig images or video. (str, default: None)')
    argparser.add_argument('-o', '--output_path',       default="./out/",                               type=str,   help='output path, str, defaults to \"output\" directory')
    argparser.add_argument('-e', '--east_path',         default="frozen_east_text_detection.pb",  type=str,   help='EAST xml filename path.')
    argparser.add_argument('-t', '--tesseract_path',    default='C:\\Program Files\\Tesseract-OCR\\tesseract.exe',  type=str,   help='Tesseract exe path.')
    argparser.add_argument('-w', '--width',             default=3200,                        type=int,   help='width.')
    argparser.add_argument('-h', '--height',            default=3200,                        type=int,   help='height.')
    argparser.add_argument('-p', '--padding',           default=0.1,                        type=float,   help='padding.')
    argparser.add_argument('-m', '--min_conf',          default=0.2,                        type=float,   help='min confidence.')
    argparser.add_argument('-l', '--log_path',          default='./new_log.log',                        type=str,   help='optional log filename.')
    
    args = argparser.parse_args()
    log_file = args.log_path
    if log_file:
        logging.basicConfig(filename = log_file,
                            filemode = 'a',
                            format = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt = '%H:%M:%S',
                            level = logging.INFO)
    else:
        logging.basicConfig(level = logging.INFO)
    
    log = logging.getLogger("opencv_haar_cascade_face_cropping_script")


    filenames = [f[:4] for f in os.listdir(args.input_path) if re.match(r'.+.tif', f)]
    dir = args.input_path#os.getcwd()+'\\input\\'
    filenames = list(set(filenames))
    log.info(f'{len(filenames)} files found total')
    res_df = pd.DataFrame()

    for i, f in enumerate(filenames):
        try:
            log.info(f'processing files {f}.tif, {f}.tfw\npair num '+str(i)+' out of '+str(len(filenames)))
            start = time.time()
            temp_res_df = process_file_by_name(dir+f, args.east_path, args.width, args.height, args.padding, args.min_conf)
            end = time.time()
            res_df = pd.concat([res_df,temp_res_df])
            log.info("Success! processing took {:.6f} seconds".format(end - start))
            if (i+1) % 10 == 0: 
                res_df.to_csv(out_path+'output_result.csv')
                log.info('\n\nSaved! Processing batch '+str(j)+' out of '+str((len(filenames)//10)+1))
                j+=1
        except Exception as err:
            log.info(f+' filename can not be processed....')
            log.info(' %s' % err)