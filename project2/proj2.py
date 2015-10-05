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
import scipy.fftpack

###############################################################################
### main 																	###
### The sequential overview of the steps to complete the assignment 		###
###############################################################################
def main(output,levels=5,fmin=.75,fmax=.85,amp=50):
	# Break up the video into jpgs
	faceVidFile = 'data/face.mp4'
	faceOutDir = 'data/face/'
	babyVidFile = 'data/baby2.mp4'
	babyOutDir = 'data/baby2/'
	faceVidSpeed = createImageLibrary(faceVidFile,faceOutDir)
	babyVidSpeed = createImageLibrary(babyVidFile,babyOutDir)

	# Laplacian Pyramids
	faceLaplacian,faceGaussian,faceImages = allThePyramids(faceOutDir,levels)
	babyLaplacian,babyGaussian,babyImages = allThePyramids(babyOutDir,levels)

	# Test a couple random image
	# print 'Displaying some example layers...'
	# print '[press esc to continue]'
	# cv2.imshow('Laplacian',faceLaplacian[5][30,:,:,:])
	# cv2.imshow('Gaussian',faceGaussian[2][222,:,:,:])
	# cv2.imshow('Laplacian 2',babyLaplacian[5][444,:,:,:])
	# cv2.imshow('Gaussian 2',babyGaussian[3][523,:,:,:])
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	print 'Filter ', fmin, fmax
	filteredFace = bandPassFilter(faceGaussian,faceVidSpeed,fmin,fmax,levels)
	filteredBaby = bandPassFilter(babyGaussian,babyVidSpeed,fmin,fmax,levels)
	print 'Amplifying the signal...'
	for j in range(levels):
		filteredFace[j] *= amp
		filteredBaby[j] *= amp

	faceOutput = output+'face_out.avi'
	babyOutput = output+'baby_out.avi'
	reconstruction(faceOutput,faceVidSpeed,faceLaplacian,filteredFace,faceGaussian,levels)
	reconstruction(babyOutput,babyVidSpeed,babyLaplacian,filteredBaby,babyGaussian,levels)
	print 'Finished at: ',faceOutput,', ',babyOutput

	# Show a bandPassed image result
	# img = filteredFace[levels-1][100].real
	# for i in range(levels):
	# 	img = cv2.pyrUp(img)
	# cv2.imshow('bandPass',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

###############################################################################
### createImageLibrary 														###
### Takes a video file and output directory, then proceeds to break up the  ###
### video into a series of jpgs by frame1 									###
###############################################################################
def createImageLibrary(video,outDir):
	vid = cv2.VideoCapture(video)
	if not os.path.isfile(outDir+'frame1.jpg'):
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

	fps = int(vid.get(cv2.CAP_PROP_FPS))
	vid.release()
	return fps

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
	gaussPyramid = [imgCopy]
	for i in range(l):
		imgCopy = cv2.pyrDown(imgCopy)
		gaussPyramid.append(imgCopy)

	# Construct the laplacian pyramid
	laplaPyramid = [gaussPyramid[-1]]
	for i in range(l,0,-1):
		# Scale up the next gaussian to do the subtraction
		nextGauss = cv2.pyrUp(gaussPyramid[i])

		# Adjust the size slightly if the dimensions became mismatched
		# during downsizing
		gs = gaussPyramid[i-1].shape

		w,h,c = nextGauss.shape
		# Subtract the two images to get the laplacian
		L = cv2.subtract(gaussPyramid[i-1][:w,:h],nextGauss[:gs[0],:gs[1]])
		laplaPyramid.append(L)

	return gaussPyramid, laplaPyramid

###############################################################################
### allThePyramids															###
### Takes in a directory location and cycles through all of the images to 	###
### create a laplacian pyramid of each one and store them intillegently		###
###############################################################################
def allThePyramids(directory,levels):
	print "Generating Laplacian Pyramids..."
	gps = []
	lps = []
	imageList = []
	once = True
	numImages = len([i for i in os.listdir(directory) if i.endswith(".jpg")])
	for j,image in enumerate(os.listdir(directory)):
		if image.endswith(".jpg"):
			gp,lp = laplacianPyramid(directory+image,levels)
			if not gp: break
			if once:
				for l in lp:
					lps.append(np.empty((numImages, l.shape[0],l.shape[1], 3),np.uint8))
				for g in gp:
					gps.append(np.empty((numImages, g.shape[0],g.shape[1], 3),np.uint8))
				once = False
			for i,level in enumerate(gp):
				gps[i][j,:,:,:] = level
			for i,level in enumerate(lp):
				lps[i][j,:,:,:] = level
			imageList.append(image)

	return lps,gps,imageList

###############################################################################
### bandPassFilter 															###
### Takes the laplacian, the frames per second, the lower cutoff freq, and  ###
### the higher cutoff freq, and returns the approximate ideal bandpass      ###
### filter.  The bandpass filter is not realistic and a butterworth would   ###
### be better																###
###############################################################################
def bandPassFilter(gaussian,framesPerSec,Fc1,Fc2,levels):
	print "Applying the band pass filter..."
	# The time between each frame, equal to the sampling rate of the fft
	samplingRate = 1.0/framesPerSec

	bandPassedResult = []
	gaussian = [i.copy() for i in gaussian]

	# Get the filtered result of the lowest level
	fft = scipy.fftpack.fft(gaussian[levels-1], axis=0)

	# The sampled frequencies of the dft
	freq = scipy.fftpack.fftfreq(gaussian[levels-1].shape[0], d=samplingRate)

	# Construct the pass as two step functions (reflective across y-axis)
	minimumFreq = (np.abs(freq-Fc1)).argmin()
	maximumFreq = (np.abs(freq-Fc2)).argmin()
	fft[:minimumFreq] = 0
	fft[maximumFreq:-maximumFreq] = 0
	fft[-minimumFreq] = 0

	# Only filter at the lowest level, and return that level plus the others
	for i in range(len(gaussian)):
		if i==levels-1:
			bandPassedResult.append(scipy.fftpack.ifft(fft,axis=0))
		else:
			bandPassedResult.append(gaussian[i])

	# Revert back to spatial domain
	return bandPassedResult

###############################################################################
### reconstruction 															###
### Takes the filtered and amplified vide in laplacian format and 			###
### reassembles it into a Video.  											###
###############################################################################
def reconstruction(output,fps,laplacian,gaussian,origGaussian,levels):
	print "Reconstructing the video..."
	l,w,h,c = origGaussian[0].shape
	vidType = cv2.VideoWriter_fourcc('M','J','P','G')
	vid = cv2.VideoWriter(output,vidType,fps,(h,w),1)
	for i in range(l):
		for j in range(levels):
			# Get the lowest level as the filtered gaussian and scale it up 
			# at every interval
			if j==0:
				imgFiltered = cv2.convertScaleAbs(gaussian[levels-1][i].real)
			else:
				imgFiltered = cv2.pyrUp(imgFiltered)

			# Take the orignal gaussian
			imgOriginal = origGaussian[levels-j-1][i]

			# And the original Laplacian
			imgLap = laplacian[j+1][i]

			# Add them together for most layers
			img = cv2.add(imgOriginal,imgLap)

			# For the important layer, add in the filtered result
			if j==levels-1:
				img = cv2.add(imgFiltered[:img.shape[0],:img.shape[1]],img[:imgFiltered.shape[0],:imgFiltered.shape[1]])
				img = cv2.convertScaleAbs(img)

		vid.write(img)
	vid.release()

if __name__ == "__main__":
	main('data/')