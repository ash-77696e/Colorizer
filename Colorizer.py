from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from basic_agent import *
import advanced_agent

'''
This is the main class file that runs the basic and advanced agent
Authors: Ashwin Haridas, Ritin Nair
'''

'''
Main function that runs our agents
'''
def main():
    run_basic_agent()
    run_advanced_agent()

'''
This functions runs our basic agent
'''
def run_basic_agent():
    # STORED AS BGR NOT RGB
    original_img = cv2.imread('smaller.jpg')
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    #print(original_img)

    left = original_img[:, 0:int(original_img.shape[1] / 2), :] # get left side
    right = original_img[:, int(original_img.shape[1] / 2):original_img.shape[1], :] # get right side
    
    # convert left and right to black and white
    left_bw = convert_to_grey(left)
    right_bw = convert_to_grey(right)

    # run k-means
    k_means_left, clusters = k_means(left)
    recolored_left = recolor_left(left, k_means_left, clusters)
    recolored_right = recolor_right(left_bw, right_bw, k_means_left, clusters)

    combined_img = []

    for i in range(0, len(left)):
        combined_img.append(list(recolored_left[i]) + list(recolored_right[i]))
    
    print(average_euclidean(recolored_right, right, right_bw))

    plt.imshow(combined_img)
    plt.show()

'''
This function runs our advanced agent
'''
def run_advanced_agent():
    # STORED AS BGR NOT RGB
    original_img = cv2.imread('original.jpg')
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    #print(original_img)

    left = original_img[:, 0:int(original_img.shape[1] / 2), :]
    right = original_img[:, int(original_img.shape[1] / 2):original_img.shape[1], :]
    
    left_bw = convert_to_grey(left)
    right_bw = convert_to_grey(right)

    # train model
    red_ws, green_ws, blue_ws = advanced_agent.train(left_bw, left)
    right_recolored = advanced_agent.test(red_ws, green_ws, blue_ws, right_bw)
    
    combined_img = []

    for i in range(0, len(left)):
        combined_img.append(list(left[i]) + list(right_recolored[i]))
    
    plt.imshow(combined_img)

    plt.show()
    print(average_euclidean(right_recolored, right, right_bw))    

'''
This function converts an image to black and white
'''
def convert_to_grey(img):
    bw = np.copy(img)
    for i in range(0, len(bw)):
        for j in range(0, len(bw[i])):
            bw[i][j] = 0.21 * bw[i][j][0] + 0.72 * bw[i][j][1] + 0.07 * bw[i][j][2]
    
    return bw

'''
This function gets the average euclidean distance of two images
'''
def average_euclidean(recolored_right, right, bw_right):
    sum = 0
    count = 0
    for i in range (bw_right[0].shape[0]):
        for j in range(bw_right[0].shape[1]):
            if i == 0 or j == 0 or i == bw_right.shape[0] - 1 or j == bw_right.shape[1] - 1: # border cell to recolor black
                continue
            sum += euclidean_distance(recolored_right[i][j], right[i][j])
            count += 1

    return sum / count


if __name__ == '__main__':
    main()