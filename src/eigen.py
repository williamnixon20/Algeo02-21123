import time
import numpy as np
from scipy.linalg import null_space
from math import sqrt
import util

def Sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def IsSymetric(m):
    return np.array_equal(m, m.transpose())

def GetFirstColVector(mat):
    # mat adalah matriks dalam numpy array, return berupa vektor kolom pertama dalam numpy array
    vector = mat[0 : len(mat), 0]
    return vector


def RoundLowerTriangularMat(mat):
    # meng-nolkan bagian segitiga bawah pada matriks, karena pada hasil matriks R di pada dekomposisi QR tidak persis 0,
    # misalnya 2.26759146e-16
    row, col = np.shape(mat)
    for i in range(row):
        for j in range(col):
            if i > j:
                mat[i, j] = 0

def GetEigenDiagonal(mat):
    eigenVals = []

    for i in range(len(mat)):
        eigenVals.append(mat[i, i])

    return eigenVals


""" QR METHOD """

def GetHouseHolder(v, iteration):
    # v adalah vektor, dalam numpy array, return berupa matriks householder dalam numpy array
    # H = I - 2(v * transpose(v))/(transpose(v) * v)
    H = np.eye(len(v)) - (2 * np.outer(v, v)) / np.linalg.norm(v) ** 2
    if iteration == 1:
        return H
    H1 = np.zeros((len(v) + 1, len(v) + 1))
    H1[0, 0] = 1
    H1[1:, 1:] = H
    return H1

def HouseholderAlgo(mat):
    # mat adalah matriks dalam numpy array, return berupa matriks Q dan R hasil dekomposisi, dengan A = QR
    A = np.copy(mat)
    R = np.copy(mat)
    row, col = np.shape(mat)
    Q = np.eye(row)
    for k in range(1, col):

        b = GetFirstColVector(A)
        e = np.zeros(len(b))

        e[0] = 1  # basis standar
        u = b + Sign(b[0]) * np.linalg.norm(b) * e
        H = GetHouseHolder(u, k)
        Q = np.matmul(Q, H.T)
        R = np.matmul(H, R)
        A = R[1:, 1:]

    RoundLowerTriangularMat(R)

    return Q, R


def QRiteration(mat, iterations):
    # Mengembalikan hasil dekomposisi QR matriks mat dengan iterasi sebanyak iteration

    Ak = np.copy(mat)
    n = mat.shape[0]
    QQ = np.eye(n)
  
    for k in range(iterations):
        # print(k+1, "iteration(s) /", iterations, "iterations")

        # if ((k+1) % 30 == 0):
        #     os.system('cls||clear')

        Q, R = HouseholderAlgo(Ak)
        QQ = np.matmul(QQ, Q)
        Ak = np.matmul(R, Q)

    # Setelah iterasi, elemen-elemen diagonal utama matriks Ak akan sama dengan eigenvalue matriks mat
    return Ak, QQ

def GetEigenValues(mat, iterations):
    # USING QR ITERATIONS
    res, V = QRiteration(mat, iterations)
    eigenVals = GetEigenDiagonal(res)

    if IsSymetric(mat):
        sortedEigenValues = np.copy(V).transpose()

        for i in range(0, len(eigenVals)):
            j = i

            sortedEigenValues[i] = sortedEigenValues[i] * Sign(sortedEigenValues[i, 0])

            while j > 0:
                if eigenVals[j] > eigenVals[j - 1]:
                    eigenVals[j], eigenVals[j - 1] = eigenVals[j - 1], eigenVals[j]
                    sortedEigenValues[[j, j - 1]] = sortedEigenValues[[j - 1, j]]
                    j -= 1

                else:
                    break

        return eigenVals, sortedEigenValues.transpose()

    return np.sort(eigenVals)[::-1], V

""" JACOBI METHOD """

def GetJacobiMax(mat):

    max = 0.0
    for i in range(len(mat) - 1):
        for j in range(i + 1, len(mat)):
            if abs(mat[i, j]) >= max:
                max = abs(mat[i, j])
                xRecord = i
                yRecord = j

    return max, xRecord, yRecord

def RotateMatrix(mat, transformationMatrix, maxAbsis, maxOrdinat):

    length = len(mat)

    diff = mat[maxOrdinat, maxOrdinat] - mat[maxAbsis, maxAbsis]
    temp = mat[maxAbsis, maxOrdinat]

    # sudut transformasi : tan (2 teta) = 2 mat[maxAbsis, maxOrdinat] / diff agar elemen max nondiagonal dijadikan nol (optimisasi iterasi)
    # jika teta sangat kecil => tan teta atau t = teta
    if abs(temp) < abs(diff) * 1.0e-30:
        t = temp / diff

    else:
        # formula tan (2 teta) ke tan teta
        cot_2_teta = diff / (2.0 * temp)
        t = (abs(cot_2_teta) + sqrt(cot_2_teta**2 + 1.0))

        if cot_2_teta > 0.0:
            t = - t

    # sin teta (s) dan cos teta (c)
    c = 1.0 / sqrt(t**2 + 1.0)
    s = t * c
    
    factor = s / (1.0 + c)

    # elemen max nondiagonal menjadi 0
    mat[maxAbsis, maxOrdinat] = 0
    
    mat[maxAbsis, maxAbsis] = mat[maxAbsis, maxAbsis] - t * temp
    mat[maxOrdinat, maxOrdinat] = mat[maxOrdinat, maxOrdinat] + t * temp

    for i in range(maxAbsis):
        temp = mat[i, maxAbsis]
        mat[i, maxAbsis] = temp - s * (mat[i, maxOrdinat] + factor * temp)
        mat[i, maxOrdinat] = mat[i, maxOrdinat] + s * (temp - factor * mat[i, maxOrdinat])

    for i in range(maxAbsis + 1, maxOrdinat):
        temp = mat[maxAbsis, i]
        mat[maxAbsis, i] = temp - s * (mat[i, maxOrdinat] + factor * mat[maxAbsis, i])
        mat[i, maxOrdinat] = mat[i, maxOrdinat] + s * (temp - factor * mat[i, maxOrdinat])

    for i in range(maxOrdinat + 1, length):
        temp = mat[maxAbsis, i]
        mat[maxAbsis, i] = temp - s * (mat[maxOrdinat, i] + factor * temp)
        mat[maxOrdinat, i] = mat[maxOrdinat, i] + s * (temp - factor * mat[maxOrdinat, i])
        
    for i in range(length):
        temp = transformationMatrix[i, maxAbsis]
        transformationMatrix[i, maxAbsis] = temp - s * (transformationMatrix[i, maxOrdinat] + factor * transformationMatrix[i, maxAbsis])
        transformationMatrix[i, maxOrdinat] = transformationMatrix[i, maxOrdinat] + s * (temp - factor * transformationMatrix[i, maxOrdinat])

def GetJacobi(covariance, threshold=1.0e-10, iterationFactor = 10):

    maxIterations = iterationFactor * (len(covariance) ** 2)
    mat = np.copy(covariance)

    transformationMatrix = np.eye(len(covariance))

    for iteration in range(maxIterations):

        max, i, j = GetJacobiMax(mat)

        if max < threshold:
            print("=====Convergence Obtained=====")
            return np.diagonal(mat), transformationMatrix

        RotateMatrix(mat, transformationMatrix, i, j)

    print("======Convergence Failure======")
    return []

def GetEigenVectors(mat, eigenVals, threshold):

    eigenVectors = []
    record = {}

    for eigenVal in eigenVals:
        if not (eigenVal in record.keys()):
            record[eigenVal] = 0

    for eigenVal in eigenVals:

        newMat = (np.eye(len(mat)) * eigenVal) - mat

        EliminateError(newMat, threshold)

        eigenVector = null_space(newMat).transpose()

        if len(eigenVector) > record[eigenVal]:

            vector = np.array(eigenVector[record[eigenVal]])

            eigenVectors.append(vector * Sign(vector[0]))
            record[eigenVal] += 1

    return np.array(eigenVectors).transpose()


def EliminateError(mat, threshold=1.0e-9):
    # eliminate small error between two similar row

    for i in range(len(mat)):
        for j in range(i + 1, len(mat)):
            cosineValue = np.dot(mat[i], mat[j]) / (
                np.linalg.norm(mat[i]) * np.linalg.norm(mat[j])
            )

            if abs(abs(cosineValue) - 1) < threshold:
                mat[j] = mat[i]



def GetEigenInfo(covariance, jacobiThreshold = 1e-10, similarityThreshold=1.0e-9):
    # eigenVals = GetEigenValues(covariance, iterations)
    eigenVals, _ = GetJacobi(covariance, jacobiThreshold)
    eigenVectors = GetEigenVectors(covariance, eigenVals, similarityThreshold)

    return (np.array(eigenVals), eigenVectors)


if __name__ == "__main__":

    # TEST CASE

    N = int(input("Masukkan N : "))
    b = np.random.randint(-1e9,1e9,size=(N,N)).astype('float64')
    b_symm = (b + b.T)/2

    print("Calculating Eigen Values : ")
    startTime = time.time()
    vals = np.sort(GetJacobi(b_symm)[0])[::-1]
    print("time :", time.time() - startTime)
    print("Our Result : ")
    print(vals)
    print("Library result : ")

    realRes = np.linalg.eig(b_symm)
    finalVal, finalVec = util.sortEigen(realRes[0], realRes[1])
    print(finalVal)

    startTime = time.time()
    vecs = GetEigenVectors(b_symm, vals, 1e-20)
    print("Calculating Eigen Vectors : ")
    print("time :", time.time() - startTime)
    print("Our Result : ")
    print(vecs)
    print("Library result : ")
    print(np.array(finalVec))



