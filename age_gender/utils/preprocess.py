import multiprocessing
import os
import yaml
import time
import cv2
import dlib
import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime
from imutils.face_utils import FaceAligner
from sklearn.model_selection import train_test_split
from utils.config_parser import get_config
FLAGS = None


def get_area(rect):
    left = rect.left()
    top = rect.top()
    right = rect.right()
    bottom = rect.bottom()
    return (bottom - top) * (right - left)


def normalize_rect(rect):
    infinity = 10 ** 10
    h, w = (infinity, infinity)
    left = rect[0] if rect[0] >= 0 else 0
    top = rect[1] if rect[1] >= 0 else 0
    right = rect[2] if rect[2] < w else w - 1
    bottom = rect[3] if rect[3] < h else h - 1
    return [left, top, right, bottom]


def scale(x_scale, y_scale, rect):
    left, top, right, bottom = rect
    weight = (right - left)/2
    height = (bottom - top)/2
    center_x = (left + right)/2
    center_y = (top + bottom)/2
    left = int(center_x - weight*x_scale)
    top = int(center_y - height*y_scale)
    right = int(center_x + weight*x_scale)
    bottom = int(center_y + height*y_scale)
    return [left, top, right, bottom]


def align_face(config, rect, face_aligner, image, gray_image):
    height_scale = config['height_scale']
    width_scale = config['width_scale']
    left = rect.left()
    top = rect.top()
    right = rect.right()
    bottom = rect.bottom()
    scaled_rect = scale(width_scale, height_scale, [left, top, right, bottom])
    normilized_rect = normalize_rect(scaled_rect)
    dlib_rect = dlib.rectangle(*normilized_rect)
    image_raw = face_aligner.align(image, gray_image, dlib_rect)
    return image_raw
