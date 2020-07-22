import math
import os
import pandas as pd
import numpy as np
from PIL import Image
from random import randint
import sys

# Extract 80x80 images of seals or background from splitted tiff files based on excel coordinates

sealWidth = 80
basewidth = 252

def mergePics(pics, decider):
	"""
	merge png parts of large tiff file if necessary
	:param pics: png parts to merge
	:param decider: specifies how the parts should be merged
	:return: single image as a result of the merging
	"""
	if (len(pics) == 1):
		return pics[0]

	if len(pics) > 2:
		result = Image.new("RGB", (2 * pics[0].width, 2 * pics[0].height))
	else:
		if not (decider[0] == 0):
			result = Image.new("RGB", (2 * pics[0].width, pics[0].height))
		else:
			result = Image.new("RGB", (pics[0].width, 2 * pics[0].height))

	if len(pics) == 2 and not (decider[1] == 0):
		result.paste(pics[0], (0,0))
		result.paste(pics[1], (0, pics[1].height))
	else:
		for i, image in enumerate(pics):
			if i < 2:
				position = (image.width * (i%2), 0)
			else:
				position = (image.width * (i%2), image.height)

			result.paste(image, position)

	return result

def isInOnePic(x, y, picWidth, picHeight):
	"""
	creates decider that specifies if and how should png parts of single tiff file be merged to extract a seal
	:param x: x coordinate of the seal
	:param y: y coordinate of the seal
	:param picWidth: width of png part
	:param picHeight: height of png part
	:return: decider that specifies if and how should png parts of single tiff file be merged to extract a seal
	"""
	curX = x - (int(x / picWidth)) * picWidth
	curY = y - (int(y / picHeight)) * picHeight
	corners = [0, 0]
	if ((curX - sealWidth / 2) < 0):
		corners[0] = -1
	if ((curY - sealWidth / 2) < 0):
		corners[1] = -1
	if ((curX + sealWidth / 2) > picWidth):
		corners[0] = 1
	if ((curY + sealWidth / 2) > picHeight):
		corners[1] = 1

	return corners

def getRequiredPics(x, y, picWidth, picHeight, overallWidth, overallHeight, path):
	"""
	gets required png parts to extract a seal
	:param x: x coordinate of the seal
	:param y: y coordinate of the seal
	:param picWidth: width of png part
	:param picHeight: height of png part
	:param overallWidth: width of tiff file
	:param overallHeight: height of tiff file
	:param path: path to png parts
	:return: list of required png parts as Image objects
	"""
	# get num of tiles in row and column
	numWTiles = math.ceil(overallWidth / picWidth)
	numHTiles = math.ceil(overallHeight / picHeight)

	# get current index
	if y < picHeight:
		index1 = 0
	else:
		index1 = int(y / picHeight)*numWTiles

	index2 = int(x / picWidth)
	curIndex = index1 + index2

	numsOfPics = [curIndex]
	decider = isInOnePic(x, y, picWidth, picHeight)

	# append adjacent pics
	if (decider[0] == 1):
		# append image from right
		numsOfPics.append(curIndex + 1)
	if (decider[0] == -1):
		# append image from left
		numsOfPics.append(curIndex - 1)
	if (decider[1] == 1):
		# append image at bottom of this one
		numsOfPics.append(curIndex + numWTiles)
		if (decider[0] == -1):
			# append image at bottom left of this one
			numsOfPics.append(curIndex + numWTiles - 1)
		if (decider[0] == 1):
			# append image at bottom right of this one
			numsOfPics.append(curIndex + numWTiles + 1)
	if (decider[1] == -1):
		# append image at top of this one
		numsOfPics.append(curIndex - numWTiles)
		if (decider[0] == -1):
			# append image at top left of this one
			numsOfPics.append(curIndex - numWTiles - 1)
		if (decider[0] == 1):
			# append image at top right of this one
			numsOfPics.append(curIndex - numWTiles + 1)

	numsOfPics.sort()
	# print(len(numsOfPics))
	imgPaths = []
	for i in numsOfPics:
		if i >= 0:
			imgPaths.append((path + "/tiles_" + str(i) + ".png"))
	pics = [Image.open(pic) for pic in imgPaths]

	return pics

def getCroppedPic(reqPics, x, y, picWidth, picHeight):
	"""
	get cropped image 80x80 of a single seal
	:param reqPics: list of required png parts to extract the seal
	:param x: x coordinate of the seal
	:param y: y coordinate of the seal
	:param picWidth: width of png part
	:param picHeight: height of png part
	:return: the seal as Image object
	"""
	decider = isInOnePic(x, y, picWidth, picHeight)
	curX = x - (int(x / picWidth)) * picWidth
	curY = y - (int(y / picHeight)) * picHeight

	if (decider[0] == -1):
		curX += picWidth
	if decider[1] == -1:
		curY += picHeight

	image = mergePics(reqPics, decider).crop((curX - sealWidth / 2, curY - sealWidth / 2, curX + sealWidth / 2, curY + sealWidth / 2))

	return image

def getBGPic(tiffPath, coords, x, y, picWidth, picHeight, overallWidth, overallHeight):
	"""
	get single 80x80 image of background that does not contain any seals
	:param tiffPath: path to png parts
	:param coords: list of seal coordinates for given tiff file
	:param x: x coordinate of the background
	:param y: y coordinate of the background
	:param picWidth: width of png part
	:param picHeight: height of png part
	:param overallWidth: width of tiff file
	:param overallHeight: height of tiff file
	:return: the background as Image object
	"""
	reversedY = overallHeight - y
	curX = x - (int(x / picWidth)) * picWidth
	curY = reversedY - (int(reversedY / picHeight)) * picHeight
	reqPics = getRequiredPics(x, reversedY, picWidth, picHeight, overallWidth, overallHeight, tiffPath)

	if (len(reqPics) == 1):
		if (not containsSeal(x, y, coords)):
			cropped = reqPics[0].crop((curX, curY, curX + sealWidth, curY + sealWidth))
			if Image.isImageType(cropped):
				cropped = cropped.convert("RGB")
				return cropped
	else:
		return None


def getSealPic(path, x, y, picWidth, picHeight, overallWidth, overallHeight):
	"""
	locate image of a seal from the given tiff file at x,y coordinates
	:param path: path to tiff file
	:param x: x coordinate
	:param y: y coordinate
	:param picWidth: width of png part
	:param picHeight: height of png part
	:param overallWidth: width of tiff file
	:param overallHeight: height of tiff file
	:return: image of a seal from the given tiff file at x,y coordinates
	"""
	y = overallHeight - y
	reqPics = getRequiredPics(x, y, picWidth, picHeight, overallWidth, overallHeight, path)
	pic = getCroppedPic(reqPics, x, y, picWidth, picHeight)

	return pic

def getSealPics(coordsPath, tiffsDirPath, sealsDirPath):
	"""
	get all 80x80 seal images from given directory of tiff directories containing the png parts
	of corresponding tiff images and store them in specified directory
	:param coordsPath: path to the excel file with the seals coordinates
	:param tiffsDirPath: path to the directory of tiff directories containing the png parts
	:param sealsDirPath: the specified directory to store the cropped images of seals
	"""
	data = pd.read_excel(coordsPath, sheet_name="PixelCoordinates")
	coords = data[['tiff_file', 'layer_name', 'x_pixel', 'y_pixel']].iterrows()
	counter = 0
	tempPath = ""
	tempTiffFile = ""
	tiffFileBase = ""
	pathExists = False

	picWidth,picHeight,overallWidth,overallHeight = 0,0,0,0

	for (_,(tiffFile, layerName, x, y)) in coords:
		if not (tempTiffFile == tiffFile):
			# set temporary temp file to that of current tiff file
			# that way the tiff dir path has to be created only once and we can also reset the counter
			tiffFileBase = os.path.splitext(tiffFile)[0]
			tiffPath = os.path.join(tiffsDirPath, tiffFileBase)
			tempTiffFile = tiffFile
			counter = 0
			pathExists = False

		if (pathExists or os.path.exists(tiffPath)):
			if (not pathExists):
				# get the current image size parameters only once
				sizeFile = open(os.path.join(tiffPath, "size.txt"), "r")
				sizes = sizeFile.read().split(',')
				(overallWidth,overallHeight,picWidth,picHeight) = [int(sizes[i]) for i in range(0,4)]
				pathExists = True

			pic = getSealPic(tiffPath, x, y, picWidth, picHeight, overallWidth, overallHeight)
			if (pic != 0):
				imgPath = os.path.join(sealsDirPath, layerName)
				if (not os.path.exists(imgPath)):
					os.mkdir(imgPath)
				pic.save(os.path.join(imgPath, tiffFileBase + "_seal" + str(counter) + ".png"))
				counter += 1
		else:
			if not (tiffPath == tempPath):
				print(tiffPath + " - Unrecognized path to tiff parts")
				tempPath = tiffPath


def getTotalXY(file, x, y, overallWidth, picWidth, picHeight):
	file = str(file)
	num = int(file[file.index("_") + 1:file.index(".")])
	numInRow = np.ceil(overallWidth / picWidth)
	toAddInCol = int(num / numInRow)
	toAddInRow = num % numInRow

	totalX = toAddInRow*picWidth + x
	totalY = toAddInCol*picHeight + y
	return totalX, totalY


def getBGPics(coordsPath, tiffsDirPath, bgDirPath, maxNum):
	"""
	get all 80x80 background images from given directory of tiff directories containing the png parts
	of corresponding tiff images and store them in specified directory
	:param coordsPath: path to the excel file with the seals coordinates
	:param tiffsDirPath: path to the directory of tiff directories containing the png parts
	:param bgDirPath: the specified directory to store the cropped images of background
	:param maxNum: max number of bg images from single tiff file
	"""
	dataCoords = pd.read_excel(coordsPath, sheet_name="PixelCoordinates")
	dataTiffs = pd.read_excel(coordsPath, sheet_name="FileOverview", usecols=315)
	allCoords = dataCoords[['tiff_file', 'x_pixel', 'y_pixel']]
	tiffNames = dataTiffs[['tile','tiff_file']].iterrows()
	maxNum = int(maxNum)

	for (_, (_,tiffName)) in tiffNames:
		print(str(tiffName))
		try:
			tiffFileBase = os.path.splitext(tiffName)[0]
		except TypeError:
			continue
		tiffPath = os.path.join(tiffsDirPath, tiffFileBase)

		if (os.path.exists(tiffPath)):
			# get the current image size parameters
			sizeFile = open(os.path.join(tiffPath, "size.txt"), "r")
			sizes = sizeFile.read().split(',')
			(overallWidth,overallHeight,picWidth,picHeight) = [int(sizes[i]) for i in range(0,4)]
			coords = allCoords[(allCoords['tiff_file'] == tiffName)].iterrows()
			numOfFiles = len(os.listdir(tiffPath))
			tiffMax = int(maxNum / numOfFiles)
			totalCounter = 0

			for file in os.listdir(tiffPath):
				if file.endswith(".txt"):
					continue
				filePath = os.path.join(tiffPath, file)
				tiffPart = Image.open(filePath)
				tiffArray = np.asarray(tiffPart)

				# if image mostly white jump over
				blacks = np.sum(tiffArray == (0,0,0))
				whites = np.sum(tiffArray == (255,255,255))

				if blacks + whites > tiffArray.size * 0.9:
					continue

				# scale the number of bg imgs required from the picture to the proportion of the picture
				# which is taken by the island
				curMax = int(tiffMax * ((blacks + whites) / tiffArray.size))
				counter = 0
				print()
				print(curMax)
				print(tiffMax)
				while counter < curMax:
					x = randint(0, picWidth - sealWidth)
					y = randint(0, picHeight - sealWidth)
					cropped = tiffArray[y:y + sealWidth, x:x + sealWidth]

					allX, allY = getTotalXY(file, x, y, overallWidth, picWidth, picHeight)

					if containsSeal(allX, allY, coords):
						continue
					# all_cropped.append(cropped)
					# cropped.save(os.path.join(bgDirPath, tiffFileBase + "_bg" + str(counter) + ".png"))
					counter += 1
					totalCounter += 1
					croppedImage = Image.fromarray(cropped)
					croppedImage.save(os.path.join(bgDirPath, tiffFileBase + "_bg" + str(totalCounter) + ".png"))

				sys.stdout.write("\r" + filePath + " - " + str(totalCounter))
				sys.stdout.flush()

				# counter = 0
				# for cropped in all_cropped:
				# 	croppedImage = Image.fromarray(cropped)
				# 	croppedImage.save(os.path.join(bgDirPath, tiffFileBase + "_bg" + str(counter) + ".png"))
				# 	counter += 1
				# 	sys.stdout.write("\r" + filePath + " - " + str(counter))
				# 	sys.stdout.flush()

		else:
			print(tiffPath + " - Unrecognized path to tiff parts")

def containsSeal(x, y, iters):
	"""
	determines if there's a seal on 80x80 image of given coordinates
	:param x: x coordinate of the image
	:param y: y coordinate of the image
	:param iters: list of seals coordinates for the given tiff file
	:return: true if seal detected, false otherwise
	"""
	for (_, (_,sealX, sealY)) in iters:
		if ((sealX <= x + sealWidth) and (sealX >= x) and (sealY >= y - sealWidth) and (sealY <= y)):
			# print(((x,y),(sealX,sealY)))
			return True

	return False

def resizePic(img, basewidth):
	"""
	resizes image
	:param img: img to resize
	:param basewidth: width to resize to
	:return: resized image
	"""
	wpercent = (basewidth/float(img.size[0]))
	hsize = int((float(img.size[1])*float(wpercent)))
	img = img.resize((basewidth,hsize), Image.ANTIALIAS)
	return img

def resizePics(dir_path, basewidth):
	"""
	resize all image in given directory
	:param dir_path:  path to the directory
	:param basewidth: width to resize to
	"""
	dir = os.listdir(dir_path)

	for image in dir:
		im = Image.open(dir_path + '/' + image)
		im = resizePic(im, basewidth)
		im.save(dir_path + '/' + image)

if __name__ == "__main__":
	sealWidth = int(sys.argv[1])
	type = sys.argv[2]
	coordsPath = sys.argv[3]
	tiffsDirPath = sys.argv[4]
	toSafeDirPath = sys.argv[5]

	if type == 'bg':
		maxNum = sys.argv[6]
		getBGPics(coordsPath, tiffsDirPath, toSafeDirPath, maxNum)
	elif (type == 'seal'):
		getSealPics(coordsPath, tiffsDirPath, toSafeDirPath)
	else:
		print("Unrecognized first argument: " + type)