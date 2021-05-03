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

def sgd(xs, ws, actual_list, alpha):
    count = 0
    initial_ws = np.copy(ws)
    while True:
        i = randint(0, xs.shape[0] - 1)
        actual_val = actual_list[i]
        old_ws = np.copy(ws)
        predicted = 255 * sigmoid(np.dot(xs[i], np.transpose(old_ws)) * 10)
        for j in range(ws.shape[1]):
            ws[0][j] = old_ws[0][j] - alpha * (2 * (predicted - actual_val) * predicted * (1 - (predicted/255)) * xs[i][j])
        count += 1

        if count == 10000:
            break
    
    return ws

def test(red_ws, green_ws, blue_ws, bw_right):
    print(red_ws)
    print(green_ws)
    print(blue_ws)
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
            
            red = 255 * sigmoid(np.dot(patch, np.transpose(red_ws)) * 10)
            green = 255 * sigmoid(np.dot(patch, np.transpose(green_ws)) * 10)
            blue = 255 * sigmoid(np.dot(patch, np.transpose(blue_ws)) * 10)
            

            recolored_right[i][j][0] = red
            recolored_right[i][j][1] = green
            recolored_right[i][j][2] = blue
            

    return recolored_right
    
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
        red_ws[0][i] = random()
        blue_ws[0][i] = random()
        green_ws[0][i] = random()

    red_ws = sgd(xs, red_ws, red_pixels, 0.001)
    blue_ws = sgd(xs, blue_ws, blue_pixels, 0.001)
    green_ws = sgd(xs, green_ws, green_pixels, 0.001)

    return red_ws, blue_ws, green_ws

if __name__ == '__main__':
    actual = np.array([1, 2, 3], dtype=float)
    print(preprocess(actual))