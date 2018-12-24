import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import floor, pi, exp
from PyQt5 import QtGui, QtCore, QtWidgets

############
# Gokay Gas
# 150150107
# 18.10.2018
############

class Window(QtWidgets.QMainWindow):

	def __init__(self):
		super(Window, self).__init__()
		self.setWindowTitle("Filtering & Geometric Transforms")
		self.setWindowState(QtCore.Qt.WindowMaximized)
    
		self.Img = None
		self.outputImg = None
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
			self.Img = cv2.imread("blocks.jpg")
			self.inputImgNo = 1

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
		return NotImplementedError

	def segmentation(self):
		if self.inputImgNo != 2:
			return
		
		grayImage = cv2.cvtColor(self.Img, cv2.COLOR_BGR2GRAY)
		ret,thresholdedImg = cv2.threshold(grayImage,56,255,cv2.THRESH_BINARY)

		erosion(15,thresholdedImg)

		R, C = thresholdedImg.shape
		qImg = QtGui.QImage(thresholdedImg.data, C, R, grayImage.strides[0], QtGui.QImage.Format_Grayscale8)
		pix = QtGui.QPixmap(qImg)
		self.label.setPixmap(pix)


#def dilation(size):

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
