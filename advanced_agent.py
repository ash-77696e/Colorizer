import numpy as np
import basic_agent
from random import *

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def preprocess(x):
    return x / 255

def get_all_patches(img):
    result = []
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            result.append(basic_agent.pixel_to_patch(img, i, j)[0])
    
    return result

def get_all_pixels(img):
    red = []
    green = []
    blue = []
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            red.append(img[i][j][0])
            green.append(img[i][j][1])
            blue.append(img[i][j][2])
    
    return red, green, blue

def sgd(xs, ws, alpha):
    pass

def train(left_bw_img, left_colored_img):
    patches = get_all_patches(left_bw_img)
    red_pixels, green_pixels, blue_pixels = get_all_pixels(left_colored_img)

    xs = np.empty((len(patches), 10))

    for i in range(xs.shape[0]):
        xs[i][0] = 1
    
    for i in range(len(patches)):
        for j in range(1, 10):
            xs[i][j] = preprocess(patches[i][j-1])
    
    red_ws = np.empty((1, 10))
    blue_ws = np.empty((1, 10))
    green_ws = np.empty((1, 10))
    
    for i in range(red_ws.shape[1]):
        red_ws[0][i] = random()
        blue_ws[0][i] = random()
        green_ws[0][i] = random()



if __name__ == '__main__':
    actual = np.array([1, 2, 3], dtype=float)
    print(preprocess(actual))