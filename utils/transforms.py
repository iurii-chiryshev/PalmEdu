import numpy as np
import torch
import random
import cv2
import math

class Rotate(object):
    def __init__(self, angle_range = (-45,45), prob = 0.5, padding_color=0):
        """

        :param angle_range: Rotation angle in degrees. Positive values mean counter-clockwise rotation.
        :param prob:
        :param padding_color:
        """
        self.arange = angle_range
        self.prob = prob
        self.padding_color = padding_color


    def __call__(self, img, label):
        if random.random() < self.prob:
            # вращаем на случайный угол относительно центра изображения
            (h, w, c) = img.shape
            angle = random.random() * (self.arange[1] - self.arange[0]) + self.arange[0]
            rot_mat = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
            img = cv2.warpAffine(img, rot_mat, (w,h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue = self.padding_color)
            label = angle + label
        return [img, label]

class VeritcalFlip(object):
    def __init__(self, prob = 0.5):
        self.prob = prob
    def __call__(self, image, label):
        if random.random() < self.prob:
            image = cv2.flip(image, 0) # veritcal flip
            label = 360 - label
        return [image, label]

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        image = image.astype(np.float32)
        for i in range(3):
            image[:,:,i] -= self.mean[i]
        for i in range(3):
            image[:,:, i] /= self.std[i]

        return [image, label]

class Resize(object):
    def __init__(self, size=(64,64)):
        self.size = size

    def __call__(self, image, label):
        image = cv2.resize(image, self.size)
        return [image, label]

class ToTensor(object):
    def __call__(self, image, label):

        image = image.transpose((2,0,1)) # change [H,W,C] to [C,H,W]

        return [image, label]

class Compose(object):
    """Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
