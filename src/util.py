import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from eigen import GetEigenInfo

HEIGHT = 254
WIDTH = 254


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
    reshapedMatriks = normalizedFaces
    return np.dot(reshapedMatriks, np.transpose(reshapedMatriks))


def GetEigenValue(T):
    # T merupakan matriks segitiga atas (tidak flatten) hasil QR algorithm
    # mengembalikan matriks 1D berisi eigen values
    eigenValues = []

    for i in range(HEIGHT):
        if T[i, i] > 0:
            eigenValues.append(T[i, i])

    return eigenValues


def GetEigenFaces(eigenVectors, normalizedFaces):
    # EigenVectors berisi seluruh matriks eigen (tidak flatten) dari semua gambar dataset,
    # normalizedFaces (flatten) berisi matriks ternormalisasi seluruh gambar dataset

    # mengembalikan array berisi eigenFaces (flatten) masing-masing gambar dataset
    importantVec = np.array(eigenVectors[1:]).transpose()
    eigenFaces = np.dot(normalizedFaces.transpose(), importantVec)
    return eigenFaces.transpose()


def sortEigen(eigenVal, eigenVec):
    tupleS = []
    for i in range(len(eigenVal)):
        tupleS.append((eigenVal[i], eigenVec[i]))
    tupleS.sort(reverse=True)
    eigenValS = []
    eigenVecS = []
    for val, vec in tupleS:
        eigenValS.append(val)
        eigenVecS.append(vec)
    return eigenValS, eigenVecS


def getWeighted(eigenFaces, normalizedData):
    ls = []
    for i in normalizedData:
        ls.append(np.dot(eigenFaces, i))

    return np.array(ls)


def getNormalizedTestImage(absPath, meanFace):
    path = absPath
    unknown_face = cv2.imread(path, 0)
    unknown_face_vector = cv2.resize(
        unknown_face, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA
    ).flatten()

    normalised_uface_vector = np.subtract(unknown_face_vector, meanFace)

    return normalised_uface_vector


def getEuclideanDistance(databaseWeighted, testWeighted):
    norms = []
    for i in range(len(databaseWeighted)):
        diff = databaseWeighted[i] - testWeighted
        norms.append(np.linalg.norm(diff, axis=1))
    return np.argmin(norms), np.min(norms)


if __name__ == "__main__":
    imagesData = GetImages(os.path.abspath("test/dataset"))
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

    (
        eigenvalues,
        eigenvectors,
    ) = GetEigenInfo(cov_matrix)

    # print(np.sort(eigenvalues))
    # print(eigenvectors)
    eigenvalues, eigenvectors = sortEigen(eigenvalues, eigenvectors)
    eigenFaces = GetEigenFaces(eigenvectors, normalizedData)
    databaseWeighted = getWeighted(eigenFaces, normalizedData)
    print(databaseWeighted.shape)

    normalizedTestImg = getNormalizedTestImage(
        os.path.abspath("test/gambar.jpg"), meanFace
    )
    testWeighted = getWeighted(eigenFaces, normalizedTestImg)
    image_index, value = getEuclideanDistance(databaseWeighted, testWeighted)
    print(value)
    img = imagesData[image_index].reshape(HEIGHT, WIDTH)
    plt.title("assoc")
    plt.imshow(img, cmap="gray")
    plt.show()
