# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse
from Tkinter import *
from tkFileDialog import askopenfilename
import Image, ImageTk
from ast import literal_eval
from scipy.spatial import Delaunay
from math import floor
import os

# The point correspondances from the GUI
points = []

###############################################################################
# takes in arguments and outputs morphed image arrays or average faces
###############################################################################
def main(faces):
	img1 = cv2.imread(faces.img1)
	img2 = cv2.imread(faces.img2)
	if img2.shape!=img1.shape:
		print 'Images need to be the same size'
		return

	# Compute the average face of the class
	if args.avg:
		averageFace('points/','15463-f15-resize/')
	else:

		# Handle the point correspondances
		img1Points = []
		img2Points = []
		# Have the user cilck points
		if not args.p1:
			print 'Select the points with left click, right click to finish'
			getPointCorrespondances(faces.img1)
			img1Points = points[:]
			del points[:]
		# Read points from given file
		else:
			f = open(args.p1)
			for line in f:
				img1Points.append(literal_eval(line))
		# img2 - clicks
		if not args.p2:
			print 'Select the points with left click, right click to finish'
			getPointCorrespondances(faces.img2)
			img2Points = points[:]
		# img2 - file
		else:
			f = open(args.p2)
			for line in f:
				img2Points.append(literal_eval(line))

		# convert to numpy arrays
		point1Set = np.array(img1Points)
		point2Set = np.array(img2Points)

		# Get the triangulations
		tri1 = Delaunay(point1Set)
		tri2 = Delaunay(point2Set)

		if not os.path.exists('output/'):
			os.makedirs('output/')

		# Morph the image
		for i in range(61):
			# Minimal Jerk Trajectory
			t = (i/61.)
			dissolveFrac = 10*t**3-15*t**4+6*t**5
			warpFrac = dissolveFrac

			morphedImg = morph(img1,img2,point1Set,point2Set,tri1,tri2,warpFrac,dissolveFrac)
			num = '00'
			if i<10:
				num = '0'+str(i)
			else:
				num = str(i)
			cv2.imwrite('output/morph'+num+'.jpg',morphedImg)
			print num

			# cv2.imshow('test',morphedImg)
			# cv2.waitKey()
			# cv2.destroyAllWindows()

###############################################################################
# The average face of a bunch of facial images can be computed.  The inputs 
# are a directory with files containing point correspondances and an image 
# directory with the images.  A points file would be 'stephen.txt' and the
# corresponding image would be 'stephen.jpg'.  
###############################################################################
def averageFace(pointsDir,imagesDir):
	# Read all of the data
	points = []
	for file in os.listdir(pointsDir):
		if file.endswith(".txt"):
			pts = []
			print file
			with open(pointsDir+file) as f:
				for line in f:
					pts.append(literal_eval(line))
			points.append((file.split('.')[0],np.array(pts).astype(np.int32)))

	# Load the first image
	imageName = imagesDir+points[0][0]+'.JPG'
	print imageName
	img = cv2.imread(imageName)
	pts1 = points[0][1]
	tri1 = Delaunay(pts1)
	points.pop(0)
	i=1
	# For each image, morph it with the current overall image by using
	# The 1/(i+1) for the ith image
	for name,pts in points:
		i+=1
		imageName = imagesDir+name+'.JPG'
		print imageName
		img2 = cv2.imread(imageName)
		tri2 = Delaunay(pts)
		img = morph(img,img2,pts1,pts,tri1,tri2,1./i,1./i).astype(np.uint8)
		pts1 = interpolate(pts1,pts,1./i)
		tri1 = Delaunay(pts1)

	cv2.imshow('test!',img)
	cv2.waitKey()
	cv2.destroyAllWindows()

	cv2.imwrite('average.jpg',img

###############################################################################
# Takes two images, corresponding feature points for both images, Delauney 
# triangulation for each image and two ratios, the ratio of the spatial 
# positioning between the two images and the ratio of the coloring between
# the two images.  It proceeds to comput the intermediate image between
# the two input images at the relative ratios using the point correspondances
# and Delauney triangulation sets.
###############################################################################
def morph(im1,im2,im1Pts,im2Pts,tri1,tri2,warpFrac,dissolveFrac):
	# Find the intermediate points between the triangle vertices
	intermediatePts = interpolate(im1Pts,im2Pts,warpFrac)
	white = (255,255,255)

	# Iterate over the triangles in the first image
	spatialWarped1 = np.zeros(im1.shape,dtype=np.uint8)
	rows,cols,channels = im1.shape
	for i in range(len(tri1.simplices)):
		# Find the ith triangle's vertices in the first image and the intermediate
		triP1 = [im1Pts[tri1.simplices[i]].astype(np.int32)]
		triPI = [intermediatePts[tri1.simplices[i]]]
		blank1 = np.zeros(im1.shape,dtype=np.uint8)
		# A mask around the triangle
		cv2.fillPoly(blank1,triP1,white)
		mask1 = cv2.bitwise_and(im1,blank1)
		# The affine transformation between the first image and the intermediate
		M1 = computeAffine(triP1[0],triPI[0])
		# The computed warp
		spatialWarped1 = warpImage(spatialWarped1,mask1,M1,triPI[0])

	# Handle the background
	_,blackMask = cv2.threshold(cv2.cvtColor(spatialWarped1,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY_INV)
	# Mask of the background region
	blackMask = cv2.cvtColor(blackMask,cv2.COLOR_GRAY2RGB)
	# The background region
	background = cv2.bitwise_and(blackMask,im1)
	# The face inserted into the background
	spatialWarped1 = background+spatialWarped1

	# Iterate over the triangles in the second image
	spatialWarped2 = np.zeros(im2.shape,dtype=np.uint8)
	rows,cols,channels = im2.shape
	for i in range(len(tri2.simplices)):
		triP2 = [im2Pts[tri2.simplices[i]]]
		triPI = [intermediatePts[tri2.simplices[i]]]
		blank2 = np.zeros(im2.shape,dtype=np.uint8)
		cv2.fillPoly(blank2,triP2,white)
		mask2 = cv2.bitwise_and(im2,blank2)
		M2 = computeAffine(triP2[0],triPI[0])
		spatialWarped2 = warpImage(spatialWarped2,mask2,M2,triPI[0])

	# Handle the background
	_,blackMask = cv2.threshold(cv2.cvtColor(spatialWarped2,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY_INV)
	blackMask = cv2.cvtColor(blackMask,cv2.COLOR_GRAY2RGB)
	background = cv2.bitwise_and(blackMask,im2)
	spatialWarped2 = background+spatialWarped2

	intermediateColors = interpolate(spatialWarped1,spatialWarped2,dissolveFrac)

	return intermediateColors

###############################################################################
# calculates the affine transformation between 3 points in one image and 3
# points in another image.  This is just a linear system of equations and 
# can be solved with Ax=b
###############################################################################
def computeAffine(p1,p2):
    assert p1.shape == p2.shape

    # Build the A matrix in Ax=b
    A = np.zeros((6,6))
    for i in range(0,6,2):
    	A[i,0:2] = p1[i/2,:]
    	A[i,2:4] = np.zeros(2)
    	A[i,4] = 1
    	A[i,5] = 0
    	A[i+1,0:2] = np.zeros(2)
    	A[i+1,2:4] = p1[i/2,:]
    	A[i+1,4] = 0
    	A[i+1,5] = 1

    # b is just all the img2 points
    b = np.reshape(p2,(6,1))
    # X = A^-1*b
    X = np.linalg.inv(A).dot(b)
    # Reshape X into 3x3 affine transform
    H = np.hstack((np.vstack((X[0:2].T,X[2:4].T)),X[4:6]))
    return H

###############################################################################
# It computse the inverse warp and figures out the pixel value in the original
# image that corresponds to each pixel in the output image.  In order to
# account for the transforms that expand the triangle, I'm filling an area
# with the value of the single pixel from the source image.  This works 
# because locations in the output that have corresponding data will get 
# overwritten with their correct value, but places that do not will get a
# nearby value - which will make the output look better
###############################################################################
def warpImage(im1,im2,M,tri):
	bottom = np.array([0,0,1])
	M = np.vstack((M,bottom))
	# inverse affine
	iM = np.linalg.inv(M)[0:2,:]
	out = im1
	# iterate over the triangular region only (not the entire image)
	# to be fair, its a minimum bounding box around the triangle...
	for y in range(int(tri[:,1].min()),int(tri[:,1].max())):
		for x in range(int(tri[:,0].min()),int(tri[:,0].max())):
			# if(out[y,x].sum()==0):
			loc = iM.dot(np.array([x,y,1]))
			try:
				if im2[loc[1],loc[0]].sum()>0:
					# Should be checking if the output pixel 
					# that I want to assign to already has 
					# content, but its super slow already...
					out[y-1,x-1] = im2[loc[1],loc[0]]
					out[y-1,x] = im2[loc[1],loc[0]]
					out[y-1,x+1] = im2[loc[1],loc[0]]
					out[y,x-1] = im2[loc[1],loc[0]]
					out[y,x] = im2[loc[1],loc[0]]
					out[y,x+1] = im2[loc[1],loc[0]]
					out[y+1,x-1] = im2[loc[1],loc[0]]
					out[y+1,x] = im2[loc[1],loc[0]]
					out[y+1,x+1] = im2[loc[1],loc[0]]
					
			except:
				pass
	return out
 	
###############################################################################
# This is very pythonic, no types, just interpolates between two things
# Handles points and images.
###############################################################################
def interpolate(p1,p2,frac):
	return p2*frac-p1*frac+p1

###############################################################################
# Handles obtaining the points in an image (lets you click through points)
# Left click for getCoords, right click for finishCoords - stores final points
# in a global because my python isn't good enough to figure out how to return
# values from an object that gets destroyed
###############################################################################
def getPointCorrespondances(imgFile):
	root = Tk()

	frame = Frame(root, bd=2, relief=SUNKEN)
	frame.grid_rowconfigure(0, weight=1)
	frame.grid_columnconfigure(0, weight=1)
	xscroll = Scrollbar(frame, orient=HORIZONTAL)
	xscroll.grid(row=1, column=0, sticky=E+W)
	yscroll = Scrollbar(frame)
	yscroll.grid(row=0, column=1, sticky=N+S)
	canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
	canvas.grid(row=0, column=0, sticky=N+S+E+W)
	xscroll.config(command=canvas.xview)
	yscroll.config(command=canvas.yview)
	frame.pack(fill=BOTH,expand=1)

	img = ImageTk.PhotoImage(Image.open(imgFile))
	canvas.create_image(0,0,image=img,anchor="nw")
	canvas.config(scrollregion=canvas.bbox(ALL))

	# Left Click
	canvas.bind("<Button-1>",getCoords)
	# Right Click
	canvas.bind("<Button-3>",lambda event: finishCoords(event,root))

	root.mainloop()

###############################################################################
# Gets an individually clicked point (left click)
###############################################################################
def getCoords(event):
	points.append((event.x,event.y))

###############################################################################
# Terminates the point clicking (right click)
###############################################################################
def finishCoords(event,root):
	root.destroy()

###############################################################################
# Argument Parsing
###############################################################################
if __name__ == "__main__":
	intro = '''Blend one face to another face using computational photography techniques'''
	parser = argparse.ArgumentParser(description=intro)
	parser.add_argument('img1', help='The first face image')
	parser.add_argument('img2', help='The second face image')
	parser.add_argument('--p1', help='Optional points file for image 1')
	parser.add_argument('--p2', help='Optional points file for image 2')
	parser.add_argument('--avg', action='store_true', help='Compute the average faces instead')
	args = parser.parse_args()
	main(args)
