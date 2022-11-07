import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

HEIGHT = 256
WIDTH = 256

def GetImages(relativePath = "test/dataset"):
    path = os.path.abspath(relativePath)

    images = [] 
    for fileName in os.listdir(path):

        imgArr = cv2.imread(os.path.join(path, fileName), 0)
        imgArr = cv2.resize(imgArr, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
        images.append(imgArr.flatten())

    return images

def GetMeanFace(images):

    # images memiliki size N data baris dan (WIDTH * HEIGHT) kolom
    meanFace = np.zeros((1,HEIGHT * WIDTH))

    for image in images:
        meanFace = np.add(meanFace, image)

    meanFace /= len(images)

    return meanFace

def GetDifference(images, meanFace):
    
    diff = images
    for i in range (len(diff)):
        diff[i] = np.subtract(diff, meanFace)
    
    return images

def GetCovariance(difference):

    return np.multiply(difference, np.transpose(difference))



# TESTING

#
# meanFace = GetMeanFace(GetImages())

# plt.imshow(meanFace.reshape(HEIGHT, WIDTH), cmap='gray')
# plt.title("Average face")
# plt.show()
