# -*- coding: utf-8 -*-
# Final Project - Stephen Smith - Computational Photography 15-463 - Fall 2015

import cv2
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import *
import argparse

def main(args):
	# Load the image
	gTiff = gdal.Open(args.geotiff)
	img = gTiff.ReadAsArray()

	print img.shape
	print img.max()
	print img.min()
	print np.median(img)

	TS = TextureSynthesis()

	# Create a mask of the no data values
	noDataMask = TS.getMask(img,-10e10)

	# Assign the no data values to a number
	img2 = TS.assignValueToMask(img,noDataMask,np.median(img))
	# print img2.dtype

	img3 = TS.textureSynthesis(img,noDataMask)

	# Transforms the image into something that can be displayed easily and shows it locally
	TS.showImage(img2,0)

	TS.showImage(img3,1)

	cv2.imwrite("textureSynthesis.png",img3)

	# Output rectified geotiff (according to initial geotiff)
	TS.outputGeoTiff(img2,'data/modified.TIF',gTiff)

class TextureSynthesis:
	edgeImg = None
	downSample = 3
	NODATATHRESHOLD = -1E10
	levels = 5
	windowSize = 3
	d = 1

	def __init__(self,levels=5,windowSize=3):
		self.levels=levels
		self.windowSize = windowSize
		self.d = (self.windowSize-1)/2

	def textureSynthesis(self,img,mask):
		out = img.copy()
		pyramidOut = self.generateGaussianPyramid(out)
		pyramidIn = self.generateGaussianPyramid(out)
		pyramidMask = self.generateGaussianPyramid(mask.astype(np.uint8))  # Does pyramid creation work for mask arrays?
		for i in range(self.downSample,self.levels):
			imSize = pyramidOut[i].shape
			self.edgeImg = np.pad(pyramidIn[i],self.d,mode='symmetric')
			for y in range(imSize[0]):
				for x in range(imSize[1]):
					if(pyramidMask[i][y,x]):
						print (y,x)
						pyramidOut[i][y,x] = self.findBestMatch(pyramidOut[i],pyramidIn[i],x,y)
		#Reconstruct Pyramid
		return reconstruct(pyramidOut)

	def findBestMatch(self,imgOut,imgIn,x,y):
		# self.edgeImg = np.pad(imgIn,self.d,mode='symmetric')
		nOut = self.buildNeighborHood(imgOut,x,y)
		nBest = np.zeros((self.windowSize,self.windowSize))
		C = None
		h,w = imgOut.shape
		for y in range(h):
			for x in range(w):
				# print 'inside', (y,x)
				nCurrent = self.buildNeighborHood(imgIn,x,y)
				if self.compareNeighborhood(nCurrent,nOut) < self.compareNeighborhood(nBest,nOut):
					nBest = nCurrent
					C = imgIn[y,x]

		return C

	def compareNeighborhood(self,nNew,nBase):
		# Handle no data inputs
		if np.any(nNew<self.NODATATHRESHOLD):
			return float("inf")
		else:
			# SSD
			return np.sum(np.square(nNew-nBase))

	def buildNeighborHood(self,img,x,y):
		# if size%2==0:
		# 	raise Exception('size must be odd')

		# h,w = img.shape

		# # Standard case, neighborhood does not include edges
		# if x>=self.d and y>=self.d and x<w-self.d and y<h-self.d:
		# 	return img[y-self.d:y+self.d+1,x-self.d:x+self.d+1]  # probably just use the below?

		# Fancy python numpy trick (really slow)
		return self.edgeImg[y:y+self.d+self.d+1,x:x+self.d+self.d+1]

	def generateGaussianPyramid(self,img):
		G = img.copy()
		pyramid = [G]
		for i in range(self.levels):
			G = cv2.pyrDown(G)
			pyramid.append(G)

		return pyramid

	def reconstruct(self,pyramid):
		reconstructed = pyramid[self.downSample]
		for i in range(self.downSample+1,self.levels):
			reconstructed = cv2.pyrUp(reconstructed)
			reconstructed = cv2.add(reconstructed,pyramid[i])

		return reconstructed

	# Returns an image array with truthiness of individual values being less than the passed value
	def getMask(self,array, value):
		return array<value

	# Takes an image and a mask and uses it to assign a value to the mask locations
	def assignValueToMask(self,array,mask,value):
		newArray = array.copy()
		newArray[mask] = value
		return newArray

	# Shows an image without modifying the original or worrying about geotiff nonsense
	def showImage(self,img,count):
		img2 = img.copy()
		img2 -= img2.min()
		img2 *= 255./(img2.max())
		img2 = cv2.resize(img2.astype(np.uint8),(500,500))
		cv2.imshow('image'+str(count),img2)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	# Saves the geotiff in the original format as a new file with the correct heading
	def outputGeoTiff(self,array,outFile,inputGeoTiff):
		# Get input information
		inputDriver = inputGeoTiff.GetDriver()
		rows = inputGeoTiff.RasterYSize
		cols = inputGeoTiff.RasterXSize

		# Start editing the output file as geotiff
		outputDriver = inputDriver.Create(outFile, cols, rows, 1, GDT_Float32)

		# Assign the array as the first band
		outBand = outputDriver.GetRasterBand(1)
		outBand.WriteArray(array,0,0)
		outBand.FlushCache()
		outBand.SetNoDataValue(-1e38)

		# Set the geo information
		outputDriver.SetGeoTransform(inputGeoTiff.GetGeoTransform())
		outputDriver.SetProjection(inputGeoTiff.GetProjection())

# Handle command line arguments
if __name__ == "__main__":
	intro = "Final Computational Photography Project by Stephen Smith - Load in a geotiff and fill the data holes"
	parser = argparse.ArgumentParser(description=intro)
	parser.add_argument('geotiff', help='The geotiff image to parse')
	args = parser.parse_args()
	main(args)