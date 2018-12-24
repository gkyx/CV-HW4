import cv2
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from math import floor, pi, exp
from PyQt5 import QtGui, QtCore, QtWidgets

############
# Gokay Gas
# 150150107
# 24.12.2018
############

class Window(QtWidgets.QMainWindow):

	def __init__(self):
		super(Window, self).__init__()
		self.setWindowTitle("Filtering & Geometric Transforms")
		self.setWindowState(QtCore.Qt.WindowMaximized)
    
		self.Img = None
		self.outputImg = None
		self.gradientArray = None
		self.cornerPoints = []
		self.inputImgNo = 0
		self.isInputOpen = False
		mainMenu = self.menuBar()

		fileMenu = mainMenu.addMenu('&File')

		# file menu actions

		openCornerAction = QtWidgets.QAction("Open Corner Detection Image", self)
		openCornerAction.triggered.connect(lambda: self.open_image(1))

		openSegAction = QtWidgets.QAction("Open Segmentation Image", self)
		openSegAction.triggered.connect(lambda: self.open_image(2))

		removeImgAction = QtWidgets.QAction("Reset", self)
		removeImgAction.triggered.connect(self.reset)

		saveAction = QtWidgets.QAction("Save", self)
		saveAction.triggered.connect(self.save_image)

		exitAction = QtWidgets.QAction("Exit", self)
		exitAction.triggered.connect(QtCore.QCoreApplication.instance().quit)

		# taskbar actions

		cornerDetectAction = QtWidgets.QAction("Run Corner Detection", self)
		cornerDetectAction.triggered.connect(self.corner_detection)

		segmentationAction = QtWidgets.QAction("Run Segmentation", self)
		segmentationAction.triggered.connect(self.segmentation)

		fileMenu.addAction(openCornerAction)
		fileMenu.addAction(openSegAction)
		fileMenu.addAction(removeImgAction)
		fileMenu.addAction(saveAction)
		fileMenu.addAction(exitAction)

		self.toolBar = self.addToolBar("ToolBar")
		self.toolBar.addAction(cornerDetectAction)
		self.toolBar.addAction(segmentationAction)

		# central widget for the opened image.
		self.centralwidget = QtWidgets.QWidget(self)
		self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
		self.horizontalLayout.setContentsMargins(100, 10, 100, 10)
		self.horizontalLayout.setSpacing(100)

		self.label = QtWidgets.QLabel(self.centralwidget)
		self.label.setAlignment(QtCore.Qt.AlignCenter)
		self.label.setStyleSheet("border:0px")
		
		self.horizontalLayout.addWidget(self.label)

		self.setCentralWidget(self.centralwidget)

		self.show()


	def open_image(self, opNo):
		if opNo == 1:
			self.Img = cv2.imread("blocks.jpg", 0)
			self.inputImgNo = 1
			
			R, C = self.Img.shape
			qImg = QtGui.QImage(self.Img.data, C, R, QtGui.QImage.Format_Grayscale8)
			pix = QtGui.QPixmap(qImg)
			self.label.setPixmap(pix)

		elif opNo == 2:
			self.Img = cv2.imread("mr.jpg")
			self.inputImgNo = 2
			
			R, C, B = self.Img.shape
			qImg = QtGui.QImage(self.Img.data, C, R, 3 * C, QtGui.QImage.Format_RGB888).rgbSwapped()
			pix = QtGui.QPixmap(qImg)
			self.label.setPixmap(pix)
		

	def reset(self):
		return NotImplementedError

	def save_image(self):
		cv2.imwrite("./output-image.png", self.outputImg)

	def corner_detection(self):
		self.gaussian_filtering(9)
		self.gradient_calculation()
		self.corner_point_search()

	def segmentation(self):
		if self.inputImgNo != 2:
			return
		
		grayImage = cv2.cvtColor(self.Img, cv2.COLOR_BGR2GRAY)
		ret,thresholdedImg = cv2.threshold(grayImage,56,255,cv2.THRESH_BINARY)

		# opening operation
		erosion(15,thresholdedImg)
		dilation(15,thresholdedImg)

		self.k_means(thresholdedImg, grayImage)

		R, C = thresholdedImg.shape
		qImg = QtGui.QImage(thresholdedImg.data, C, R, grayImage.strides[0], QtGui.QImage.Format_Grayscale8)
		pix = QtGui.QPixmap(qImg)
		self.label.setPixmap(pix)

	def k_means(self, maskedRegion, originalImage):
		lastCenters = [0,0]
		#centers = [random.randint(0,255), random.randint(0,255)]  # When they are random it does not always give the optimum one.
		centers = [173, 248] # gives the result that we want
		points = []
		#clusters = [[],[]]
		clusterTotals = [0,0]
		clusterLengths = [0,0]

		while lastCenters != centers:
			print("Last Centers: ", lastCenters[0], " - ", lastCenters[1])
			print("Centers: ", centers[0], " - ", centers[1])
			clusterTotals = [0,0]
			clusterLengths = [0,0]

			for i in range(maskedRegion.shape[0]):
				for j in range(maskedRegion.shape[1]):
					if maskedRegion[i,j] == 255:
						points.append([i,j])

			for point in points:
				if abs(originalImage[point[0],point[1]] - centers[0]) <= abs(originalImage[point[0],point[1]] - centers[1]):
					#clusters[0].append(point)
					clusterTotals[0] += originalImage[point[0],point[1]]
					clusterLengths[0] += 1
				else:
					#clusters[1].append(point)
					clusterTotals[1] += originalImage[point[0],point[1]]
					clusterLengths[1] += 1

			lastCenters[0] = centers[0]
			lastCenters[1] = centers[1]
			centers[0] = clusterTotals[0] // clusterLengths[0]
			centers[1] = clusterTotals[1] // clusterLengths[1]
		
		for point in points:
			if abs(originalImage[point[0],point[1]] - centers[0]) <= abs(originalImage[point[0],point[1]] - centers[1]):
				maskedRegion[point[0],point[1]] = 127
			else:
				maskedRegion[point[0],point[1]] = 255

	def gaussian_filtering(self, size):
		standardDeviation = 0.01 * size

		self.outputImg = np.zeros([self.Img.shape[0], self.Img.shape[1]], dtype=np.uint8)

		# prepare the gaussian kernel
		kernel = np.zeros([size, size], dtype='int64')
		for i in range(size):
			for j in range(size):
				kernel[i,j] = round((1 / (2 * pi * standardDeviation)) * exp(-((((i - (size // 2))*(i - (size // 2))) + ((j - (size // 2))*(j - (size // 2)))) / (2 * standardDeviation * standardDeviation))) * 100)

		expandedImage = np.zeros([self.Img.shape[0] + 2 * floor(size / 2), self.Img.shape[1] + 2 * floor(size / 2)], dtype=np.uint8)
		expandedImage[floor(size / 2):(-floor(size / 2)),floor(size / 2):(-floor(size / 2))] = self.Img

		for i in range(self.Img.shape[0]):
			for j in range(self.Img.shape[1]):
				self.outputImg[i,j] = np.sum(np.sum((kernel*expandedImage[i:i+size, j:j+size]),0),0) // np.sum(np.sum(kernel, 0), 0)

		# show the outputted image
		#R, C = self.outputImg.shape
		#qImg = QtGui.QImage(self.outputImg.data, C, R, QtGui.QImage.Format_Grayscale8)
		#pix = QtGui.QPixmap(qImg)
		#self.label.setPixmap(pix)

	def gradient_calculation(self):
		self.gradientArray = np.zeros([self.outputImg.shape[0], self.outputImg.shape[1], 2])
		
		
		for i in range(self.outputImg.shape[0]):
			for j in range(self.outputImg.shape[1]):
				if j == 0 or j == self.outputImg.shape[1] - 1 or i == 0 or i == self.outputImg.shape[0] - 1:
					self.gradientArray[i,j,:] = 0
				else:
					self.gradientArray[i,j,0] = (float(self.outputImg[i, j + 1]) - float(self.outputImg[i, j - 1])) / 2 # Ix
					self.gradientArray[i,j,1] = (float(self.outputImg[i + 1, j]) - float(self.outputImg[i - 1, j])) / 2 # Iy

	def corner_point_search(self):
		size = 3   # size of the Window
		expandedGradient = np.zeros([self.outputImg.shape[0] + 2 * floor(size / 2), self.outputImg.shape[1] + 2 * floor(size / 2), 2], dtype="int64")
		expandedGradient[floor(size / 2):(-floor(size / 2)),floor(size / 2):(-floor(size / 2)), :] = self.gradientArray

		for i in range(self.outputImg.shape[0]):
			for j in range(self.outputImg.shape[1]):
				gArray = np.zeros([2,2])
				gArray[0,0], gArray[1,1] = np.sum(np.sum(np.multiply(expandedGradient[i:i+size, j:j+size],expandedGradient[i:i+size, j:j+size]),0),0) # Ix^2 and Iy^2 are calculated.
				gArray[0,1] = np.sum(np.sum(np.multiply(expandedGradient[i:i+size, j:j+size, 0], expandedGradient[i:i+size, j:j+size, 1]),0),0)
				gArray[1,0] = gArray[0,1]
				
				if (np.linalg.eigvals(gArray).min() > 600):
					self.cornerPoints.append((j,i))
		
		backtorgb = cv2.cvtColor(self.Img, cv2.COLOR_GRAY2RGB)

		for point in self.cornerPoints:
			self.draw_point(backtorgb, point, (0,255,0))

		R, C, B = backtorgb.shape
		qImg = QtGui.QImage(backtorgb.data, C, R, 3 * C, QtGui.QImage.Format_RGB888).rgbSwapped()
		pix = QtGui.QPixmap(qImg)
		self.label.setPixmap(pix)

	def draw_point(self, img, p, color ) :
		cv2.circle( img, p, 2, color, cv2.FILLED, cv2.LINE_AA, 0 )


def dilation(size, img):
	#structElement = np.zeros([size,size], dtype=np.uint8)
	#structElement[:,:] = 1

	expandedImage = np.zeros([img.shape[0] + 2 * floor(size / 2), img.shape[1] + 2 * floor(size / 2)], dtype="int64")
	expandedImage[floor(size / 2):(-floor(size / 2)),floor(size / 2):(-floor(size / 2))] = img

	for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				# since the struct element only consists of 1s we do not need to use it we can just check all the values in the window(frame)
				if np.sum(np.sum(expandedImage[i:i+size, j:j+size] == 255,0),0) > 0:
					img[i,j] = 255
				else:
					img[i,j] = 0

def erosion(size, img):
	#structElement = np.zeros([size,size], dtype=np.uint8)
	#structElement[:,:] = 1

	expandedImage = np.zeros([img.shape[0] + 2 * floor(size / 2), img.shape[1] + 2 * floor(size / 2)], dtype="int64")
	expandedImage[floor(size / 2):(-floor(size / 2)),floor(size / 2):(-floor(size / 2))] = img

	for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				# since the struct element only consists of 1s we do not need to use it we can just check all the values in the window(frame)
				if np.sum(np.sum(expandedImage[i:i+size, j:j+size] == 255,0),0) == size**2:
					img[i,j] = 255
				else:
					img[i,j] = 0


def main():
	app = QtWidgets.QApplication(sys.argv)
	GUI = Window()
	sys.exit(app.exec_())

main()
