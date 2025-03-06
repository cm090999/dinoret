from typing import Any
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
import time

class PILToNumpy(object):
    def __call__(self, img):
        # Convert the PIL image to a NumPy array
        img = np.array(img)
        return img
    
class NumpyToPIL(object):
    def __call__(self, img: np.ndarray):
        # Convert the Numpy image to a PIL array
        img = Image.fromarray(img)
        return img
    

class CropRoundImage(object):
    def __call__(self, img: np.ndarray):

        # start = time.time()

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Threshold the image to create a binary mask
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find the coordinates of the white areas
        y_nonzero, x_nonzero = np.where(thresh)

        # Calculate the bounding box of the white areas
        min_y, max_y = np.min(y_nonzero), np.max(y_nonzero)
        min_x, max_x = np.min(x_nonzero), np.max(x_nonzero)

        # Crop the image to the bounding box
        cropped_image = img[min_y:max_y+1, min_x:max_x+1]

        # end = time.time()
        # print("CropRoundImage: ", end - start)

        return cropped_image
    
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image: np.ndarray):

        # start = time.time()

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))
        equalized_image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # end = time.time()
        # print("CLAHETransform: ", end - start)

        return equalized_image
    
class MedianFilterTransform:
    def __init__(self, filter_size=3):
        self.filter_size = filter_size

    def __call__(self, pil_image):

        # start = time.time()

        filtered_pil_imgae = cv2.medianBlur(pil_image, self.filter_size)

        # end = time.time()
        # print("MedianFilterTransform: ", end - start)
    
        return filtered_pil_imgae