from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from basic_agent import *

def main():
    # STORED AS BGR NOT RGB
    original_img = cv2.imread('image.png')
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    #print(original_img)

    left = original_img[:, 0:int(original_img.shape[1] / 2), :]
    right = original_img[:, int(original_img.shape[1] / 2):original_img.shape[1], :]
    
    # left_bw = convert_to_grey(left)
    # right_bw = convert_to_grey(right)
    
    # combined_img = []

    # for i in range(0, len(left)):
    #     combined_img.append(list(left[i]) + list(right[i]))
    
    # plt.imshow(combined_img)

    # plt.show()

    k_means_left, clusters = k_means(left)
    recolored_left = recolor_left(left, k_means_left, clusters)

    combined_img = []

    for i in range(0, len(left)):
        combined_img.append(list(recolored_left[i]) + list(right[i]))

    plt.imshow(combined_img)
    plt.show()

def convert_to_grey(img):
    bw = np.copy(img)
    for i in range(0, len(bw)):
        for j in range(0, len(bw[i])):
            bw[i][j] = 0.21 * bw[i][j][0] + 0.72 * bw[i][j][1] + 0.07 * bw[i][j][2]
    
    return bw


if __name__ == '__main__':
    main()