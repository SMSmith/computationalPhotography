# -*- coding: utf-8 -*-
###############################################################################################
### Stephen Smith 																			###	
### 9-29-15																					###
### This software serves as a demonstration of Eulerian Video Magnification 				###
### It was written for assignment for 15-463 Computational Photography at   				###
### CMU.  The results will be displayed here:												###
### https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15463-f15-users/smsmith/proj2/www/	###
###############################################################################################

import cv2
import os
import numpy as np

###############################################################################
### main 																	###
### The sequential overview of the steps to complete the assignment 		###
###############################################################################
def main(levels=6):
	# Break up the video into jpgs
	faceVidFile = 'data/face.mp4'
	faceOutDir = 'data/face/'
	babyVidFile = 'data/baby2.mp4'
	babyOutDir = 'data/baby2/'
	createImageLibrary(faceVidFile,faceOutDir)
	createImageLibrary(babyVidFile,babyOutDir)

	# Laplacian Pyramids
	faceLaplacian,faceGaussian,faceImages = allThePyramids(faceOutDir,levels)
	babyLaplacian,babyGaussian,babyImages = allThePyramids(babyOutDir,levels)

	# Test a couple random image
	cv2.imshow('Laplacian',faceLaplacian[0][30])
	cv2.imshow('Gaussian',faceGaussian[2][222])
	cv2.imshow('Laplacian 2',babyLaplacian[5][444])
	cv2.imshow('Gaussian 2',babyGaussian[3][523])
	cv2.waitKey(0)
	cv2.destroyAllWindows()

###############################################################################
### createImageLibrary 														###
### Takes a video file and output directory, then proceeds to break up the  ###
### video into a series of jpgs by frame1 									###
###############################################################################
def createImageLibrary(video,outDir):
	if not os.path.isfile(outDir+'frame1.jpg'):
		vid = cv2.VideoCapture(video)
		imgNum = 0
		success = True
		while success:
			success, image = vid.read()
			if not success:
				break
			cv2.imwrite(outDir+"frame%d.jpg"%imgNum,image)
			if cv2.waitKey(10) == 27:
				break
			imgNum+=1
		vid.release()

###############################################################################
### laplacianPyramid														###
### Takes in an image and returns a laplacian pyramid of the image with l	###
### levels																	###
###############################################################################
def laplacianPyramid(image,l):
	# Read in the image and make a copy to downsample
	img = cv2.imread(image)
	imgCopy = img.copy()

	# Construct the gaussian pyramid
	gaussPyramid = []
	for i in range(l):
		imgCopy = cv2.pyrDown(imgCopy)
		gaussPyramid.append(imgCopy)

	# Construct the laplacian pyramid
	laplaPyramid = [gaussPyramid[-1]]
	for i in range(l-1,0,-1):
		# Scale up the next gaussian to do the subtraction
		nextGauss = cv2.pyrUp(gaussPyramid[i])

		# Adjust the size slightly if the dimensions became mismatched
		# during downsizing
		gs = gaussPyramid[i-1].shape
		if gs!=nextGauss.shape:
			nextGauss = cv2.resize(nextGauss,(gs[1],gs[0]))

		# Subtract the two images to get the laplacian
		L = cv2.subtract(gaussPyramid[i-1],nextGauss)
		laplaPyramid.append(L)

	return gaussPyramid, laplaPyramid

###############################################################################
### allThePyramids															###
### Takes in a directory location and cycles through all of the images to 	###
### create a laplacian pyramid of each one and store them intillegently		###
###############################################################################
def allThePyramids(directory,levels):
	print "Generating Laplacian Pyramids..."
	gps = [[] for i in range(levels)]
	lps = [[] for i in range(levels)]
	imageList = []
	for image in os.listdir(directory):
		if image.endswith(".jpg"):
			gp,lp = laplacianPyramid(directory+image,levels)
			if not gp: break
			for i,level in enumerate(gp):
				gps[i].append(level)
			for i,level in enumerate(lp):
				lps[i].append(level)
			imageList.append(image)

	return lps,gps,imageList

if __name__ == "__main__":
	main()