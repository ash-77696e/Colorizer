import numpy as np
from random import *

class Point:
    def __init__(self, rgb, clusterNum):
        self.rgb = rgb
        self.clusterNum = clusterNum

def euclidean_distance(a, b):
    return np.linalg.norm(b - a)

def k_means(img):
    imgArr = np.empty((img.shape[0], img.shape[1]), dtype = Point)
    clusters = []
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            imgArr[i][j] = Point(img[i][j], -1)
        
    for i in range(5):
        x = randint(0, img.shape[0] - 1)
        y = randint(0, img.shape[1] - 1)
        clusters.append(img[x][y])
        
    while True:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                shortest_distance = 500
                cluster = 0

                for x in range(5):
                    euclidean = euclidean_distance(imgArr[i][j].rgb, clusters[x])
                    if euclidean < shortest_distance:
                        shortest_distance = euclidean
                        cluster = x
                
                imgArr[i][j].clusterNum = cluster

        oldCenters = []

        for x in range(5):
            numPoints = 0
            sumR = 0
            sumG = 0
            sumB = 0

            for i in range(imgArr.shape[0]):
                for j in range(imgArr.shape[1]):
                    if imgArr[i][j].clusterNum == x:
                        sumR += imgArr[i][j].rgb[0]
                        sumG += imgArr[i][j].rgb[1]
                        sumB += imgArr[i][j].rgb[2]
                        numPoints += 1
            
            oldCenters.append(clusters[x])
            
            if numPoints != 0:
                clusters[x][0] = sumR / numPoints
                clusters[x][1] = sumG / numPoints
                clusters[x][2] = sumB / numPoints
        
        oldCtr = 0

        for i in range(5):
            if abs(oldCenters[i][0] - clusters[i][0]) < 0.1:
                oldCtr += 1
            if abs(oldCenters[i][1] - clusters[i][1]) < 0.1:
                oldCtr += 1
            if abs(oldCenters[i][2] - clusters[i][2]) < 0.1:
                oldCtr += 1
        

        if oldCtr == 15:
            break
            
        
    
    return imgArr, clusters

def recolor_left(left, k_means_left, clusters):
    recolored_left = np.copy(left)
    for i in range(left.shape[0]):
        for j in range(left.shape[1]):
            cluster = k_means_left[i][j].clusterNum
            recolored_left[i][j] = clusters[cluster]
    
    return recolored_left