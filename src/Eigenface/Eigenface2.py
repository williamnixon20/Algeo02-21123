import numpy as np

def Sign(x) :
    if x>=0 : 
        return 1
    else :
        return -1

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
    # Mengembalikan hasil dekomposisi QR matriks mat dengan iterasi sebanyak iterations
    row = np.shape(mat)[0]
    vals = []
    Ak = np.copy(mat)
    n = mat.shape[0]
    QQ = np.eye(n)
    for k in range(iterations) :
        Q, R = HouseholderAlgo(Ak)
        QQ = np.matmul(Q, Q)
        Ak = np.matmul(R, Q)
    
    # Setelah iterasi, elemen-elemen diagonal utama matriks Ak akan sama dengan eigenvalue matriks mat
    return Ak
    
if __name__ == "__main__" :
    vals = []
    # TEST CASE
    #test = np.array([[1,2,3],[1,1,1],[2,1,3]])
    #test = np.array([[3,5,2,1,7], [7,6,9,0,4], [2,3,1,7,5], [8,9,6,5,2], [2,3,4,5,6]])
    test = np.array([[3,-2,0], [-2,3,0], [0,0,5]])
    #test = np.array([[-1,4,-2], [-3,4,0], [-3,1,3]])
    #test = np.array([[0,0,-2], [1,2,3], [1,0,3]])
    print("Matriks : ")
    print(test)
    print()

    res = QRiteration(test, 20) #percobaan dengan iterasi 20 kali
    for i in range(len(test)) :
        vals.append(res[i,i])
    
    print("Nilai eigen: ")
    print(vals)

    print("Nilai eigen (dengan library numpy): ")
    print(np.linalg.eigvals(test))
    