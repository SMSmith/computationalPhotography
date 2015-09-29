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
def main():
	faceVidFile = 'data/face2.mp4'
	faceOutDir = 'face/'
	babyVidFile = 'data/baby2.mp4'
	babyOutDir = 'baby/'
	if not os.path.isfile(faceOutDir+'frame1.jpg'):
		createImageLibrary(faceVidFile,faceOutDir)
	if not os.path.isfile(babyOutDir+'frame1.jpg'):
		createImageLibrary(babyVidFile,babyOutDir)

###############################################################################
### createImageLibrary 														###
### Takes a video file and output directory, then proceeds to break up the  ###
### video into a series of jpgs by frame1 									###
###############################################################################
def createImageLibrary(video,outDir):
	vid = cv2.VideoCapture(video)
	imgNum = 0
	while success:
		success, image = vid.read()
		cv2.imwrite(outDir+"frame%d.jpg"%imgNum,image)
		if cv2.waitkey(10) == 27:
			break
		imgNum+=1

if __name__ == "__main__":
	main()