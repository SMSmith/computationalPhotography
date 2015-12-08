# -*- coding: utf-8 -*-
# Project 6 - Stephen Smith - Computational Photography 15-463 - Fall 2015

import cv2
import numpy as np
import argparse
from Tkinter import *
from tkFileDialog import askopenfilename
import Image, ImageTk

points = []
canvas = None

def main(args):
	img1 = cv2.imread(args.img1)
	img2 = cv2.imread(args.img2)
	# img1 = cv2.resize(img1,(0,0),fx=0.2,fy=0.2)
	# img2 = cv2.resize(img2,(0,0),fx=0.2,fy=0.2)

	if args.click:
		getPointCorrespondances(args.img1)
		img1Points = np.array(list(points)).T
		del points[:]
		getPointCorrespondances(args.img2)
		img2Points = np.array(list(points)).T
		del points[:]
	else:
		# room
		# img1Points = np.array([[2999,3047,3201,3001,3277],[3867,3871,4088,4065,4095]])
		# img2Points = np.array([[460,497,638,493,702],[127,125,304,284,312]])
		# buffalo
		img1Points = np.array([[114,283,336,390,609,609],[122,124,110,128,130,184]])
		img2Points = np.array([[349,503,552,603,843,845],[127,128,110,126,119,176]])

	print img1Points
	print img2Points

	M = computeH(img1Points,img2Points)
	# print cv2.getPerspectiveTransform(img1Points,img2Points)
	print M
	h2,w2,_ = img2.shape
	h1,w1,_ = img1.shape

	# Open CV Comparison
	img3 = cv2.warpPerspective(img2,M,(w2,h2))
	# My warp
	img4,dimensions1 = warpImage(img2,M,(w2,h2))
	img5,dimensions2 = warpImage(img1,np.identity(3),(w1,h1))
	
	# img4 = cv2.resize(img4,(w,h))
	cv2.imshow('original',img2)
	cv2.imshow('OCV_warped',img3)
	cv2.imshow('my_warp',img4)
	# cv2.imshow('blended',img6)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imwrite('data/myWarp.jpg',img4)
	cv2.imwrite('data/ocvWarp.jpg',img3)

	img6 = blend(img4,img5,dimensions1,dimensions2)
	cv2.imshow('blended',img6)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imwrite('data/blended.jpg',img6)

def blend(img1,img2,dimensions1,dimensions2):
	w = max(max(max(dimensions1[0][1]-dimensions1[0][0],dimensions2[0][1]-dimensions2[0][0]),dimensions2[0][1]-dimensions1[0][0]),dimensions1[0][1]-dimensions2[0][0])
	h = max(max(max(dimensions1[1][1]-dimensions1[1][0],dimensions2[1][1]-dimensions2[1][0]),dimensions2[1][1]-dimensions1[1][0]),dimensions1[1][1]-dimensions2[1][0])
	print dimensions1, dimensions2
	print h,w

	# shift = np.identity(3)
	shiftX = dimensions1[0][0]
	shiftY = dimensions1[1][0]
	print shiftX,shiftY

	print (h,w)

	shiftedImg = np.zeros((h,w,3),np.uint8)
	h2,w2,_ = img2.shape
	h1,w1,_ = img1.shape
	print h2,w2,h1,w1

	# Gauss Pyramid
	levels = 5
	im1 = img1.copy()
	im2 = img2.copy()
	gaus1 = [im1]
	gaus2 = [im2]
	for level in range(levels-1):
		im1 = cv2.pyrDown(im1)
		gaus1.append(im1)
		im2 = cv2.pyrDown(im2)
		gaus2.append(im2)
		cv2.imshow('test',im2)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	# Laplacian Pyramid
	lapla1 = [gaus1[-1]]
	lapla2 = [gaus2[-1]]
	for i in range(levels-1,0,-1):
		GE = cv2.pyrUp(gaus1[i])
		# print GE.shape, gaus1[i-1].shape, GE[0:gaus1[i-1].shape[0],0:gaus1[i-1].shape[1]].shape
		L = cv2.subtract(gaus1[i-1],GE[0:gaus1[i-1].shape[0],0:gaus1[i-1].shape[1]])
		lapla1.append(L)
		GE = cv2.pyrUp(gaus2[i])
		L = cv2.subtract(gaus2[i-1],GE[0:gaus2[i-1].shape[0],0:gaus2[i-1].shape[1]])
		lapla2.append(L)
		cv2.imshow('test',L)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	pyramid = []
	for level in range(levels):
		im1 = lapla1[levels-level-1]
		im2 = lapla2[levels-level-1]
		h2,w2,_ = im2.shape
		h1,w1,_ = im1.shape
		if shiftX>0 and shiftY>0:
			shiftedImg[shiftY:h1+shiftY,shiftX:w1+shiftX] = im1
			for x in range(w2):
				for y in range(h2):
					if np.sum(shiftedImg[y,x])==0:
						shiftedImg[y,x] = im2[y,x]
					else:
						shiftedImg[y,x] = 0.5*shiftedImg[y,x]+0.5*im2[y,x]
		elif shiftY>0:
			shiftedImg[shiftY:h1+shiftY,0:w1] = im1
			for x in range(-shiftX,w2-shiftX):
				for y in range(h2):
					if np.sum(shiftedImg[y,x])==0:
						shiftedImg[y,x] = im2[y,x+shiftX]
					else:
						shiftedImg[y,x] = 0.5*shiftedImg[y,x]+0.5*im2[y,x+shiftX]
		elif shiftX>0:
			shiftedImg[0:h1,shiftX:w1+shiftX] = im1
			for x in range(w2):
				for y in range(-shiftY,h2-shiftY):
					if np.sum(shiftedImg[y,x])==0:
						shiftedImg[y,x] = im2[y+shiftY,x]
					else:
						shiftedImg[y,x] = 0.5*shiftedImg[y,x]+0.5*im2[y+shiftY,x]
		else:
			shiftedImg[0:h1,0:w1] = im1
			for x in range(-shiftX,w2-shiftX):
				for y in range(-shiftY,h2-shiftY):
					if np.sum(shiftedImg[y,x])==0:
						shiftedImg[y,x] = im2[y+shiftY,x+shiftX]
					else:
						shiftedImg[y,x] = 0.5*shiftedImg[y,x]+0.5*im2[y+shiftY,x+shiftX]
		pyramid.append(shiftedImg)
		cv2.imshow('test',shiftedImg)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		h = int(h/2)+1
		w = int(w/2)+1
		shiftedImg = np.zeros((h,w,3),np.uint8)
		shiftY = int(shiftY/2)
		shiftX = int(shiftX/2)

	# Reconstruct
	out = pyramid[levels-1]
	for i in range(levels-2,-1,-1):
		out = cv2.pyrUp(out)
		if sum(pyramid[i].shape) < sum(out.shape):
			size = pyramid[i].shape
		else:
			size = out.shape
		out = cv2.add(out[0:size[0],0:size[1]],pyramid[i][0:size[0],0:size[1]])

	return out


###############################################################################
# Warps the image by the given transformation 								  #
###############################################################################
def warpImage(img,M,size):
	w,h = size
	print (w,h)

	# Use the inverse warp method
	iM = np.linalg.inv(M)
	
	# Make the image the correct size
	corners = []
	x,y = (0,0)
	corners.append(np.array([(M[0,0]*x+M[0,1]*y+M[0,2])/(M[2,0]*x+M[2,1]*y+M[2,2]),(M[1,0]*x+M[1,1]*y+M[1,2])/(M[2,0]*x+M[2,1]*y+M[2,2])]))
	x,y = (w,0)
	corners.append(np.array([(M[0,0]*x+M[0,1]*y+M[0,2])/(M[2,0]*x+M[2,1]*y+M[2,2]),(M[1,0]*x+M[1,1]*y+M[1,2])/(M[2,0]*x+M[2,1]*y+M[2,2])]))
	x,y = (w,h)
	corners.append(np.array([(M[0,0]*x+M[0,1]*y+M[0,2])/(M[2,0]*x+M[2,1]*y+M[2,2]),(M[1,0]*x+M[1,1]*y+M[1,2])/(M[2,0]*x+M[2,1]*y+M[2,2])]))
	x,y = (0,h)
	corners.append(np.array([(M[0,0]*x+M[0,1]*y+M[0,2])/(M[2,0]*x+M[2,1]*y+M[2,2]),(M[1,0]*x+M[1,1]*y+M[1,2])/(M[2,0]*x+M[2,1]*y+M[2,2])]))

	maxW = int(corners[0][0])
	maxH = int(corners[0][1])
	minW = int(corners[0][0])
	minH = int(corners[0][1])
	for c in corners:
		if c[0] > maxW:
			maxW = int(c[0])
		if c[1] > maxH:
			maxH = int(c[1])
		if c[0] < minW:
			minW = int(c[0])
		if c[1] < minH:
			minH = int(c[1])

	w,h = (maxW-minW,maxH-minH)
	print (w,h)

	# blank image
	out = np.zeros((h,w,3),np.uint8)

	if np.array_equal(M,np.identity(3)):
		out = img
		return (out,[(minW,maxW),(minH,maxH)])

	# Solve for the warped image
	for y in range(minH,maxH):
		for x in range(minW,maxW):
			# Makes sure the image is mapped to the correct location
			loc = np.array([(iM[0,0]*x+iM[0,1]*y+iM[0,2])/(iM[2,0]*x+iM[2,1]*y+iM[2,2]),(iM[1,0]*x+iM[1,1]*y+iM[1,2])/(iM[2,0]*x+iM[2,1]*y+iM[2,2])])
			# loc = iM.dot(np.array([x,y,1]))
			bot = np.floor(loc)
			top = np.ceil(loc)
			ratio = (loc-bot)

			if(loc[0]<0 or loc[1]<0 or loc[0]>=w or loc[1]>=h):
				continue
			try:
				botL = img[bot[1],bot[0]]
				botR = img[bot[1],top[0]]
				topL = img[top[1],bot[0]]
				topR = img[top[1],top[0]]

				# Interp between 4 pixels
				out[y-minH,x-minW] = (botL*(1-ratio[0])+botR*ratio[0])*(1-ratio[1])+(topL*(1-ratio[0])+topR*ratio[0])*ratio[1]
			except:
				pass

	return (out,[(minW,maxW),(minH,maxH)])


###############################################################################
# calculates the affine transformation between two sets of points 			  #
###############################################################################
def computeH(p1,p2):
	assert p1.shape == p2.shape

	N = p1.shape[1]

	# Build the A matrix in Ax=b
	p1x = p1[0:1,:]
	p1y = p1[1:,:]
	p2x = p2[0:1,:]
	p2y = p2[1:,:]
	p2xy = np.concatenate((p2x,p2y,np.ones((1,N))))
	z3 = np.zeros((3,N))
	row1 = np.concatenate((p2xy,z3),axis=1)
	row2 = np.concatenate((z3,p2xy),axis=1)
	row3 = np.concatenate((np.multiply(-p1x,p2x),np.multiply(-p1y,p2x)),axis=1)
	row4 = np.concatenate((np.multiply(-p1x,p2y),np.multiply(-p1y,p2y)),axis=1)
	row5 = np.concatenate((-p1x,-p1y),axis=1)

	A = np.concatenate((row1,row2,row3,row4,row5))
	A = A.T

	U,_,_ = np.linalg.svd(np.dot(A.T,A))

	H = np.reshape(U[:,8],(3,3))
	return H
	# return H

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

if __name__=="__main__":
	intro = "Image warping and mosaicing"
	parser = argparse.ArgumentParser(description=intro)
	parser.add_argument('img1', help='The left image')
	parser.add_argument('img2', help='The right image')
	parser.add_argument('--click','-c', action='store_true', help='Get points by clicking')
	args = parser.parse_args()
	main(args)