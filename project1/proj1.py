# -*- coding: utf-8 -*-

import cv2
import os
import tarfile
import glob
import numpy as np

def openCVExperiments():
	# Extract the data
	workingDirectory = "C:/Users/Stephen/Documents/compPhoto/project1/"
	if not os.path.exists(workingDirectory):
		os.makedirs(workingDirectory)
	if not os.path.exists(workingDirectory+'00029u.tif'):
		tar = tarfile.open("C:/Users/Stephen/Downloads/images.tar")
		tar.extractall(path=workingDirectory)
		tar.close()
		
	# Open Image 1 for viewing (messing with opencv)
	img1 = cv2.imread('00087u.tif')
	#cv2.imshow('Image 1',img1)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	
	# Downsize image 1 (messing with opencv)
	downSize = 0.1
	img1Small = cv2.resize(img1,(0,0),fx=downSize,fy=downSize)
	cv2.imshow('Image 1 Small', img1Small)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	
	# Crop image 1 (messing with opencv)
	h,w,c = img1.shape
	cropAmount = 110
	img1Cropped = img1[cropAmount:h-cropAmount,cropAmount:w-cropAmount]
	img1CroppedSmall = cv2.resize(img1Cropped,(0,0),fx=downSize,fy=downSize)
	#cv2.imshow('Image 1 Cropped, Downsized',img1CroppedSmall)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
		
	# Divide Image 1 (messing with opencv)
	H = h/3
	W = w
	img1_1 = cv2.resize(img1[0:H,0:W],(0,0),fx=downSize,fy=downSize)
	img1_2 = cv2.resize(img1[H+1:2*H,0:W],(0,0),fx=downSize,fy=downSize)
	img1_3 = cv2.resize(img1[2*H+1:3*H,0:W],(0,0),fx=downSize,fy=downSize)
	#cv2.imshow('Image 1/3',img1_1)
	#cv2.imshow('Image 2/3',img1_2)
	#cv2.imshow('Image 3/3',img1_3)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	# Find the borders (messing with opencv)
	darkRows = ()
	darkBorders = ()
	for i in range(h):
		averageBrightness = np.sum(img1[i,0:W])/W/3
		if averageBrightness < 42:
			darkRows+=i,
			# The current row is dark, but the previosu row is not
			if not i-1 in darkRows:
				darkBorders+=i,
		else:
			# The current row is light, but the previous row is dark
			if i-1 in darkRows:
				darkBorders+=i-1,
	dividingRows = ()
	for i in range(0,len(darkBorders),2):
		start = darkBorders[i]
		end = darkBorders[i+1]
		average = (start+end)/2
		dividingRows+=average,
		
	print dividingRows
	
def main(maxShift=15,startScale=.0625,endScale=1.0):
	# Load the images
	images = ()
	for file in glob.glob('0*.jpg'):
		images += cv2.imread(file),
	# Extra images
	for file in glob.glob('extra/*'):
		images += cv2.imread(file),

	# The number of pixels to shift each channel of each image
	minAlignments = []

	# For every image in the data set
	for i,img in enumerate(images):
		h,w,c = img.shape
		tempScale = startScale
		theEndScale = endScale
		# Crop some border
		# img = img[100:h]
		# h,w,c=img.shape

		# The jpg images are smaller and don't need more than two pyramid levels
		if h < 2000:
			tempScale = 1.0
			theEndScale = 1.0

		# Double the scaling factor each time until it exceeds 1
		newImage = True
		tempShift = maxShift
		while tempScale <= theEndScale:
			tempImg = cv2.resize(img,(0,0),fx=tempScale,fy=tempScale)
			tempScale*=2
			h,w,c = tempImg.shape
			H = h/3

			# Image Division (3 parts for each channel)
			channels = [tempImg[n*H:(n+1)*H-1,0:w] for n in range(3)]
			baseChannel = channels[0]

			# for c,channel in enumerate(channels):
			# 	cv2.imshow('gray '+str(c),channel)

			# Align the channels
			bestAlignment = (0,0)
			if newImage:
				# Start with no shift for blue and maxshift (center) for green and red (it gets multiplied by 2)
				minAlignments.append([(0,0),(maxShift/2.,maxShift/2.),(maxShift/2.,maxShift/2.)]) #[Image: (Blue: ),(Green: ),(Red: )]
				newImage = False
			for c,channel in enumerate(channels[1:]):
				minScore = float('inf')
				# The alignments of the last pyramid level converted to this level
				prevAlignVer = int(minAlignments[-1][c+1][0]*2)
				prevAlignHor = int(minAlignments[-1][c+1][1]*2)
				# Center on previous alignment, range is +/- maxShift
				for j in range(prevAlignVer-maxShift,prevAlignVer+maxShift):
					for k in range(prevAlignHor-maxShift,prevAlignHor+maxShift):
						if j < 0: continue
						if k < 0: continue
						if j >= 2*tempShift: continue
						if k >= 2*tempShift: continue
						# print j,k,H,w,tempShift, channel.shape
						score = np.sum(np.sum(np.square(channel[j:H-2*tempShift+j,k:w-2*tempShift+k]-baseChannel[tempShift:H-tempShift,tempShift:w-tempShift])))
						if score < minScore:
							minScore = score 	
							bestAlignment = (j,k)
				minAlignments[-1][c+1] = bestAlignment
			if h<2000:
				minAlignments[-1][0] = (int(2*theEndScale/tempScale*maxShift),int(2*theEndScale/tempScale*maxShift))
			else:
				minAlignments[-1][0] = (tempShift,tempShift)
			tempShift*=2

		# BGR!!!  Colorize
		tempShift/=2
		coloredImg = baseChannel[tempShift:H-tempShift,tempShift:w-tempShift]
		for c,channel in enumerate(channels):
			colorMap = cv2.cvtColor(channel, cv2.COLOR_BGR2GRAY)
			shift = minAlignments[-1][c]
			ver = shift[0]
			hor = shift[1]

			coloredImg[:,:,c] = colorMap[ver:H-2*tempShift+ver,hor:w-2*tempShift+hor]
			# blankImage = np.zeros((coloredImg.shape), np.uint8)
			# blankImage[:,:,c] = colorMap[ver:H-2*tempShift+ver,hor:w-2*tempShift+hor]
			# cv2.imshow('Color'+str(c),blankImage)

		# Make the image a usable size and display it
		h,_,_ = coloredImg.shape
		if h>2000:
			coloredDownSizedImg = cv2.resize(coloredImg,(0,0),fx=500./h,fy=500./h)
		else:
			coloredDownSizedImg = coloredImg
		# cv2.imshow('Image_e Aligned '+str(i), coloredDownSizedImg)
		# cv2.waitKey()
		# cv2.destroyAllWindows()
		cv2.imwrite("JPG_Image_{}_colorized.jpg".format(i),coloredImg)
		print [i-15 for b in minAlignments[-1] for i in b]

	print minAlignments
	
if __name__ == "__main__":
	#openCVExperiments()
	main(maxShift=15,startScale=.125,endScale=1)