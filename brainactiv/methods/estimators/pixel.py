import numpy as np
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy


def compute_warmth(img):
	hue = np.array(img.convert("HSV"))[:,:,[0]]
	saturation = np.array(img.convert("HSV"))[:,:,[1]]
	value = np.array(img.convert("HSV"))[:,:,[2]]
	measure = np.cos(hue/255*np.pi*2) * (saturation / 255) * (value / 255)
	measure = ((measure + 1) / 2)
	measure = np.moveaxis(measure, -1, 0)
	return measure

def compute_saturation(img):
	measure = np.array(img.convert("HSV"))[:,:,[1]]
	measure = np.moveaxis(measure, -1, 0) / 255
	return measure

def compute_brightness(img):
	measure = np.array(img.convert("HSV"))[:,:,[2]]
	measure = np.moveaxis(measure, -1, 0) / 255
	return measure

def compute_entropy(img):
	image = img_as_ubyte(np.array(img.convert("L")))
	measure = entropy(image, disk(5)) / np.log2(256)
	measure = measure[None,:,:]
	return measure
