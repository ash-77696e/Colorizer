import numpy as np
import basic_agent
from random import *

'''
This is the advanced agent class which has functions necessary for the advanced agent's implementation
Authors: Ashwin Haridas, Ritin Nair
'''

'''
This function applies the sigmoid function to a given value(s)
'''
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

'''
This function preprocesses a given value(s)
'''
def preprocess(x):
    return x / 255

'''
This function returns a list of all patches in an image
'''
def get_all_patches(img):
    result = []
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            result.append(basic_agent.pixel_to_patch(img, i, j)[0])
    
    return result

'''
This function returns a list of all pixels in an image
'''
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

'''
This function determines the loss of our model
'''
def loss(xs, ws, actual_list):
    sum = 0
    for i in range(xs.shape[0]):
        predicted = 255*sigmoid(np.dot(xs[i], np.transpose(ws)))
        actual = actual_list[i]
        sum += (predicted - actual) ** 2
    return sum

'''
This function runs stocastic gradient descent to train our model
'''
def sgd(xs, ws, actual_list, alpha):
    count = 0
    initial_ws = np.copy(ws)
    while True:
        i = randint(0, xs.shape[0] - 1)
        actual_val = actual_list[i]
        old_ws = np.copy(ws)
        predicted = 255* sigmoid(np.dot(xs[i], np.transpose(old_ws))) # get predicted
        # print(loss(xs, ws, actual_list))
        for j in range(ws.shape[1]):
            ws[0][j] = old_ws[0][j] - alpha * (2 * (predicted - actual_val) * predicted * (1 - predicted/255) * xs[i][j]) # update equation
        count += 1

        if count == 10000:
            break
    return ws

'''
This function recolors the right side
'''
def test(red_ws, green_ws, blue_ws, bw_right):
    recolored_right = np.copy(bw_right)
    for i in range (bw_right.shape[0]):
        for j in range(bw_right.shape[1]):
            if i == 0 or i == bw_right.shape[0] - 1 or j == 0 or j == bw_right.shape[1] - 1:
                recolored_right[i][j][0] = 0
                recolored_right[i][j][1] = 0
                recolored_right[i][j][2] = 0
                continue
            curr_patch, pixel_indices = basic_agent.pixel_to_patch(bw_right,i, j)
            patch = np.empty((1, 10))
            for k in range(10):
                if k == 0:
                    patch[0][k] = 1
                else:
                    patch[0][k] = curr_patch[k - 1] / 255
            
            red = 255*sigmoid(np.dot(patch, np.transpose(red_ws))) # apply model with appropriate weights
            green = 255*sigmoid(np.dot(patch, np.transpose(green_ws)))
            blue = 255*sigmoid(np.dot(patch, np.transpose(blue_ws)))
            

            recolored_right[i][j][0] = red
            recolored_right[i][j][1] = green
            recolored_right[i][j][2] = blue
            

    return recolored_right

'''
This function sets up the model to train
'''    
def train(left_bw_img, left_colored_img):
    patches = get_all_patches(left_bw_img)
    red_pixels, green_pixels, blue_pixels = get_all_pixels(left_colored_img)

    xs = np.empty((len(patches), 10), dtype=float)

    for i in range(xs.shape[0]):
        xs[i][0] = 1
    
    for i in range(len(patches)):
        for j in range(1, 10):
            xs[i][j] = preprocess(patches[i][j-1])
    
    red_ws = np.empty((1, 10), dtype=float)
    blue_ws = np.empty((1, 10), dtype=float)
    green_ws = np.empty((1, 10), dtype=float)
    
    for i in range(red_ws.shape[1]):
        red_ws[0][i] = 0.0005
        blue_ws[0][i] = 0.0005
        green_ws[0][i] = 0.0005

    alpha = 0.00005

    red_ws = sgd(xs, red_ws, red_pixels, alpha)
    blue_ws = sgd(xs, blue_ws, blue_pixels, alpha)
    green_ws = sgd(xs, green_ws, green_pixels, alpha)

    return red_ws, green_ws, blue_ws
