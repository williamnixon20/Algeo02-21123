import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from face_detector import facecropImage
from eigen import GetEigenInfo

HEIGHT = 256
WIDTH = 256


def GetImagesNorm(absPath):
    # mengembalikan matriks seluruh gambar dataset berukuran HEIGH * WIDTH (flatten)
    path = absPath

    images = []
    for fileName in os.listdir(path):

        imgArr = cv2.imread(os.path.join(path, fileName), 0)
        images.append(imgArr)

    return images


def GetImagesTrain(absPath):
    # mengembalikan matriks seluruh gambar dataset berukuran HEIGH * WIDTH (flatten)
    path = absPath

    images = []
    for fileName in os.listdir(path):

        imgArr = cv2.imread(os.path.join(path, fileName), 0)
        imgArr = cv2.resize(imgArr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        images.append(imgArr.flatten())

    return images


def GetMeanFace(images):

    # images memiliki size N data baris dan (WIDTH * HEIGHT) kolom (flatten)
    # mengembalikan matriks rata-rata (flatten)
    meanFace = np.zeros((1, HEIGHT * WIDTH))

    for image in images:
        meanFace = np.add(meanFace, image)

    meanFace /= len(images)

    return meanFace


def GetNormalized(images, meanFace):
    # images berisi matriks (flatten) seluruh gambar dataset, meanFace berisi matriks (flatten) rata-rata seluruh gambar dataset
    # mengembalikan matriks-matriks ternormalisasi (flatten)
    normalizedFaces = np.ndarray(shape=(len(images), HEIGHT * WIDTH))
    for i in range(len(images)):
        normalizedFaces[i] = np.subtract(images[i], meanFace)

    return normalizedFaces


def GetCovariance(normalizedFaces):
    # normalizedFaces berisi matriks ternormalisasi (flatten) seluruh gambar dataset
    # mengembalikan matriks kovarian (tidak flatten)
    # reshapedMatriks = np.reshape(normalizedFaces, (HEIGHT, WIDTH))

    return np.matmul(normalizedFaces, np.transpose(normalizedFaces))


# def GetEigenFaces(eigenVectors, normalizedFaces):
#     # EigenVectors berisi seluruh matriks eigen (tidak flatten) dari semua gambar dataset,
#     # normalizedFaces (flatten) berisi matriks ternormalisasi seluruh gambar dataset

#     # mengembalikan array berisi eigenFaces (flatten) masing-masing gambar dataset
#     importantVec = np.array(eigenVectors).transpose()
#     # print(importantVec)
#     importantVec = importantVec[1:]
#     print(importantVec)
#     eigenFaces = np.dot(normalizedFaces.transpose(), importantVec.transpose())
#     print(eigenFaces.shape)
#     return eigenFaces.transpose()


def sortEigen(eigenVal, eigenVec):
    tupleS = []
    vecTranspose = np.transpose(eigenVec)
    for i in range(len(eigenVal)):
        tupleS.append((eigenVal[i], vecTranspose[i]))
    tupleS.sort(reverse=True)
    eigenValS = []
    eigenVecS = []
    for val, vec in tupleS:
        eigenValS.append(val)
        eigenVecS.append(vec)
    return eigenValS, np.transpose(eigenVecS)


# def getWeighted(eigenFaces, normalizedData):
#     ls = []
#     for i in normalizedData:
#         ls.append(np.matmul(eigenFaces, i))

#     return np.array(ls)

def getEigenFaces(eigenVectors, covariance):

    filteredVectors = np.transpose(eigenVectors)[1:]
    return np.matmul(covariance, np.transpose(filteredVectors))

def getTestEigenFaces(eigenVectors, normalizedFaces, testNormalized):
    
    filteredVectors = np.transpose(eigenVectors)[1:]
    expandedVectors = np.matmul(np.transpose(normalizedFaces), np.transpose(filteredVectors))

    return np.matmul(testNormalized, expandedVectors)

def getNormalizedTestImage(absPath, meanFace, intellicrop = True):
    path = absPath
    unknown_face = cv2.imread(path, 0)

    if (intellicrop):
        unknown_face = facecropImage(unknown_face)
        
    unknown_face_vector = cv2.resize(
        unknown_face, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA
    ).flatten()

    normalised_uface_vector = np.subtract(unknown_face_vector, meanFace)

    return normalised_uface_vector


def getEuclideanDistance(databaseWeighted, testWeighted):
    norms = []
    print(testWeighted)
    for i in range(len(databaseWeighted)):
        diff = databaseWeighted[i] - testWeighted
        norms.append(np.sqrt(np.sum((diff) ** 2)))
    return np.argmin(norms), np.min(norms)


if __name__ == "__main__":
    imagesData = GetImagesTrain(os.path.abspath("test/dataset_itb"))
    meanFace = GetMeanFace(imagesData)
    normalizedData = GetNormalized(imagesData, meanFace)
    cov_matrix = GetCovariance(normalizedData)

    # (
    #     eigenvalues,
    #     eigenvectors,
    # ) = np.linalg.eig(cov_matrix)
    # print(np.sort(eigenvalues))
    # print(eigenvectors)

    # eigenvalues, eigenvectors = sortEigen(eigenvalues, eigenvectors)
    # eigenFaces = GetEigenFaces(eigenvectors, normalizedData)
    print((cov_matrix == cov_matrix.T).all())
    print(np.linalg.norm(cov_matrix - cov_matrix.T) < 1e-8)
    (
        eigenvalues,
        eigenvectors,
    ) = GetEigenInfo(cov_matrix)


    # print(np.sort(eigenvalues))
    # print(eigenvectors)
    # eigenvalues, eigenvectors = sortEigen(eigenvalues, eigenvectors)
    # eigenFaces = GetEigenFaces(eigenvectors, normalizedData)
    # databaseWeighted = getWeighted(eigenFaces, normalizedData)
    # print(databaseWeighted.shape)

    # normalizedTestImg = getNormalizedTestImage(
    #     os.path.abspath("test/gambar.jpg"), meanFace
    # )
    # testWeighted = getWeighted(eigenFaces, normalizedTestImg)
    # image_index, value = getEuclideanDistance(databaseWeighted, testWeighted)
    # print(value)
    # img = imagesData[image_index].reshape(HEIGHT, WIDTH)
    # plt.title("assoc")
    # plt.imshow(img, cmap="gray")
    # plt.show()


