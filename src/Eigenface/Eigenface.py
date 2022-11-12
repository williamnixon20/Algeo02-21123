import os

import cv2
import numpy as np
from scipy.linalg import null_space

HEIGHT = 256
WIDTH = 256

def Sign(x) :
    if x>=0 : 
        return 1
    else :
        return -1

def IsSymetric(m):
    return np.array_equal(m, m.transpose())

def GetHouseHolder(v, iteration) :
    # v adalah vektor, dalam numpy array, return berupa matriks householder dalam numpy array
    # H = I - 2(v * transpose(v))/(transpose(v) * v)
    H = np.eye(len(v)) - (2 * np.outer(v, v))/np.linalg.norm(v)**2
    if iteration == 1 :
        return H
    H1 = np.zeros((len(v)+1, len(v)+1))
    H1[0,0] = 1
    H1[1:, 1:] = H
    return H1

def GetFirstColVector(mat) :
    # mat adalah matriks dalam numpy array, return berupa vektor kolom pertama dalam numpy array
    vector = mat[0:len(mat), 0]
    return vector

def RoundLowerTriangularMat(mat) :
    # meng-nolkan bagian segitiga bawah pada matriks, karena pada hasil matriks R di pada dekomposisi QR tidak persis 0, 
    # misalnya 2.26759146e-16
    row, col = np.shape(mat)
    for i in range(row) :
        for j in range(col) :
            if (i>j) :
                mat[i, j] = 0

def HouseholderAlgo(mat) :
    # mat adalah matriks dalam numpy array, return berupa matriks Q dan R hasil dekomposisi, dengan A = QR
    A = np.copy(mat)
    R = np.copy(mat)
    row, col = np.shape(mat)
    Q = np.eye(row)
    for k in range(1, col) :
        #print("k =", k)
        b = GetFirstColVector(A)
        #print("b =", b)
        e = np.zeros(len(b)); e[0] = 1 # basis standar
        u = b + Sign(b[0]) * np.linalg.norm(b) * e
        #print("u =", u)
        H = GetHouseHolder(u, k)
        Q = np.matmul(Q, H.T)
        #print("H = ")
        #print(H)
        R = np.matmul(H, R)
        #print("R =")
        #print(R)
        A = R[1:, 1:]
        #print("A' =")
        #print(A)
    RoundLowerTriangularMat(R)
    
    return Q, R

def QRiteration(mat, iterations) :
    # Mengembalikan hasil dekomposisi QR matriks mat dengan iterasi sebanyak iteration
    

    Ak = np.copy(mat)
    n = mat.shape[0]
    QQ = np.eye(n)
    print("=============Entering QR interation=============")
    for k in range(iterations) :
        # print(k+1, "iteration(s) /", iterations, "iterations")

        # if ((k+1) % 30 == 0):
        #     os.system('cls||clear')

        Q, R = HouseholderAlgo(Ak)
        QQ = np.matmul(QQ, Q)
        Ak = np.matmul(R, Q)
    
    # Setelah iterasi, elemen-elemen diagonal utama matriks Ak akan sama dengan eigenvalue matriks mat
    return Ak, QQ

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
        normalizedFaces[i] = np.subtract(normalizedFaces[i], meanFace)
    
    return images

def GetCovariance(normalizedFaces):
    # normalizedFaces berisi matriks ternormalisasi (flatten) seluruh gambar dataset
    # mengembalikan matriks kovarian (tidak flatten)
    # reshapedMatriks = np.reshape(normalizedFaces, len(normalizedFaces), HEIGHT, WIDTH)

    return np.matmul(normalizedFaces , np.transpose(normalizedFaces))


def GetEigenDiagonal(mat):
    eigenVals = []

    for i in range(len(mat)) :
        eigenVals.append(mat[i,i])

    return eigenVals

def GetEigenValues(mat, iterations):
    res, V = QRiteration(mat, iterations)
    eigenVals= GetEigenDiagonal(res)

    if (IsSymetric(mat)):
        sortedEigenValues = np.copy(V).transpose()
        
        for i in range(0, len(eigenVals)):
            j = i

            sortedEigenValues[i] = sortedEigenValues[i] * Sign(sortedEigenValues[i, 0])

            while(j > 0):
                if (eigenVals[j] > eigenVals[j-1]):
                    eigenVals[j], eigenVals[j-1] = eigenVals[j-1], eigenVals[j]
                    sortedEigenValues[[j, j-1]] = sortedEigenValues[[j-1, j]]
                    j -= 1

                else:
                    break

        return eigenVals, sortedEigenValues.transpose()

    return np.sort(eigenVals)[::-1], V

def GetEigenVectors(mat, eigenVals) :
    
    eigenVectors = []
    record = {}

    for eigenVal in eigenVals:
        if (not(eigenVal in record.keys())):
            record[eigenVal] = 0

    for eigenVal in eigenVals:

        newMat = (np.eye(len(mat)) * eigenVal) - mat
        # print(newMat)
        EliminateError(newMat, 1e-9)
       
        eigenVector = null_space(newMat).transpose()

        if (len(eigenVector) > record[eigenVal]):

            vector = np.array(eigenVector[record[eigenVal]])
            print(vector)
            eigenVectors.append(vector *  Sign(vector[0]))
            record[eigenVal] += 1
        

    return np.array(eigenVectors).transpose()


def EliminateError(mat, threshold = 1e-9):
    # eliminate small error between two similar row

    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            cosineValue = np.dot(mat[i], mat[j]) / (np.linalg.norm(mat[i]) * np.linalg.norm(mat[j]))
        
            if (abs(abs(cosineValue) - 1) < threshold):
                mat[j] = mat[i]

def GetEigenInfo(covariance, iterations = 10000):
    eigenVals, V = GetEigenValues(covariance, iterations)
    eigenVectors = GetEigenVectors(covariance, eigenVals)

    return [np.array(eigenVals), eigenVectors]

if __name__ == "__main__" :

    # TEST CASE
    test = np.array([[1,2,3],[1,1,1],[2,1,3]])
    # test = np.array([[3,5,2,1,7], [7,6,9,0,4], [2,3,1,7,5], [8,9,6,5,2], [2,3,4,5,6]])
    # test = np.array([[3,-2,0], [-2,3,0], [0,0,5]])
    # test = np.array([[-1,4,-2], [-3,4,0], [-3,1,3]])
    # test = np.array([[0,0,-2], [1,2,3], [1,0,3]])

    # test = np.array([[3,6, 7], [6, 7, 8], [7, 8, 9]])
    # test = np.array([[complex(1, 1), 0], [complex(1, 1), 0]])
    print("Matriks : ")
    print(test)
    
    # eigenVals, V = GetEigenValues(test, 10000)

    # if (IsSymetric(test)):
    #     eigenVectors = V

    # else:
    #     eigenVectors = GetEigenVectors(test, eigenVals)
    # print("Nilai eigen: ")
    # print(eigenVals)
    # print("Eigen vectors: ")
    # print(eigenVectors)

    # print("Nilai eigen (dengan library numpy): ")
    # print(np.linalg.eig(test))

    # (
    # eigenvalues,
    # eigenvectors,
    # ) = np.linalg.eig(test)
    # print(eigenvalues)
    
    print(GetEigenInfo(test))

    # print("dari lib :", (eigenvectors))
    #DATA SET TESTING
    # ProccessDataset(100)

    # MEAN FACE TESTING

    #
    # meanFace = GetMeanFace(GetImages())

    # plt.imshow(meanFace.reshape(HEIGHT, WIDTH), cmap='gray')
    # plt.title("Average face")
    # plt.show()

