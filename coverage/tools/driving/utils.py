# import cv2
import math
import random
from collections import defaultdict

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
# from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.preprocessing import image


def preprocess_image(img_path, target_size=(100, 100)):
    img = image.load_img(img_path, target_size=target_size)
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess_input(input_img_data)
    return input_img_data


def preprocess_image_chauffeur(img_path, target_size=(120, 320)):
    import cv2
    # TODO: too many hard-code. this function is only used for driving-chauffeur
    img = cv2.imread(img_path)
    img = cv2.resize(img, (320, 240))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = img[120:240, :, :]
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img = ((img - (255.0 / 2)) / 255.0)
    img = img.reshape(1, 120, 320, 3)
    return img


def deprocess_image(x):
    x = x.reshape((100, 100, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def atan_layer(x):
    return tf.multiply(tf.atan(x), 2)


def atan_layer_shape(input_shape):
    return input_shape


def normal_init(shape):
    return K.truncated_normal(shape, stddev=0.1)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

