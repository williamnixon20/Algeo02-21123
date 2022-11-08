import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

HEIGHT = 256
WIDTH = 256

def GetImages(relativePath = "test/dataset"):
    # mengembalikan matriks seluruh gambar dataset berukuran HEIGH * WIDTH (flatten)
    path = os.path.abspath(relativePath)

    images = [] 
    for fileName in os.listdir(path):

        imgArr = cv2.imread(os.path.join(path, fileName), 0)
        imgArr = cv2.resize(imgArr, (WIDTH, HEIGHT), interpolation = cv2.INTER_AREA)
        images.append(imgArr.flatten())

    return images

def GetMeanFace(images):

    # images memiliki size N data baris dan (WIDTH * HEIGHT) kolom (flatten)
    # mengembalikan matriks rata-rata (flatten)
    meanFace = np.zeros((1,HEIGHT * WIDTH))

    for image in images:
        meanFace = np.add(meanFace, image)

    meanFace /= len(images)

    return meanFace

def GetNormalized(images, meanFace):
    # images berisi matriks (flatten) seluruh gambar dataset, meanFace berisi matriks (flatten) rata-rata seluruh gambar dataset
    # mengembalikan matriks-matriks ternormalisasi (flatten)
    normalizedFaces = np.copy(images)
    for i in range (len(normalizedFaces)):
        normalizedFaces[i] = np.subtract(normalizedFaces, meanFace)
    
    return images

def GetCovariance(normalizedFaces):
    # normalizedFaces berisi matriks ternormalisasi (flatten) seluruh gambar dataset
    # mengembalikan matriks kovarian (tidak flatten)
    reshapedMatriks = np.reshape(len(reshapedMatriks), HEIGHT, WIDTH)
    return np.multiply(reshapedMatriks , np.transpose(reshapedMatriks))

def GetEigenValue(T):
    # T merupakan matriks segitiga atas (tidak flatten) hasil QR algorithm
    # mengembalikan matriks 1D berisi eigen values
    eigenValues = []

    for i in range(HEIGHT):
        if (T[i, i] > 0):
            eigenValues.append(T[i, i])
    
    return eigenValues

def GetEigenFaces(EigenVectors, NormalizedFaces):
    # EigenVectors berisi seluruh matriks eigen (tidak flatten) dari semua gambar dataset, 
    # normalizedFaces (flatten) berisi matriks ternormalisasi seluruh gambar dataset

    # mengembalikan array berisi eigenFaces (flatten) masing-masing gambar dataset
    EigenFaces = (np.copy(NormalizedFaces)).reshaped(len(NormalizedFaces), HEIGHT, WIDTH)

    for i in range(len(EigenFaces)):
        EigenFaces[i] = (np.multiply(EigenVectors, EigenFaces[i])).flatten()

    return EigenFaces

# TESTING

#
# meanFace = GetMeanFace(GetImages())

# plt.imshow(meanFace.reshape(HEIGHT, WIDTH), cmap='gray')
# plt.title("Average face")
# plt.show()
