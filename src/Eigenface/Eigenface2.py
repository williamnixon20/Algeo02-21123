import numpy as np

def GetColVector(mat, col, startRow) :
    Vector = [0 for i in range(len(mat))]
    for i in range(startRow-1, len(mat)) :
        Vector[i] = mat[i][col-1]
    return Vector

def GetQR(mat) : # Householder algorithm
    R = mat
    Q = np.identity(len(mat))
    for i in range(1, len(R[0])) :
        x = GetColVector(R, i, i)
        # print("X = ")
        # print(x)
        x[i-1] -= np.linalg.norm(x)
        x /= np.linalg.norm(x)
        # print("V = ")
        # print(x)
        H = np.identity(len(x)) - 2*np.outer(x,x)
        Q = np.matmul(Q, H)
        R = np.matmul(H, R) 
        # print("H = ")
        # print(H)
        # print("Q = ")
        # print(Q)
        # print("R = ")
        # print(R)
    return Q, R

#testcase
if __name__ == "__main__" :
    # x = [[1,-1,4], [1,4,-2], [1,4,2], [1,-1,0]]
    x = np.array([[-1,-1,1], [1,3,3], [-1,-1,5]])
    Q, R = GetQR(x)
    print("x = ")
    print(x)
    print("Q = ")
    print(Q)
    print("R = ")
    print(R)
    print("QxR = ")
    print(np.matmul(Q, R))