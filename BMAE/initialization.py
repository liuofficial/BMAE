import scipy.linalg as spl
import numpy as np
from loadhsi import loadhsi

def initial(E,  alpha, mu, lamb, kernel_class):

    [UF, SF] = spl.svd(np.dot(E.T, E))[: 2]
    IF = np.dot(np.dot(UF, np.diag(1. / (SF + 2))), UF.T)
    w1_init = np.dot(E, IF).T

    b1_init = IF

    w2_init = alpha / (alpha + mu)
    c2_init = mu / (alpha + mu)
    b2_init = c2_init * E


    [UF2, SF2] = spl.svd((1 - alpha) * matrix_kernel(E, E, kernel_class))[: 2]

    IF2 = np.dot(np.dot(UF2, np.diag(1. / (SF2 + mu))), UF2.T)
    w3_init = (IF2 * mu).T
    b3_init = (1 - alpha) * w3_init.T

    c3_init = E
    w4_init = lamb / mu
    w5_init = E

    return w1_init,b1_init,  w2_init, c2_init, b2_init, w3_init, b3_init, c3_init, w4_init, w5_init


def rbf_kernel(x1, x2, sigma=2.0):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))


def matrix_kernel(matrix1, matrix2, kernel_class):
    rows1, cols1 = matrix1.shape
    rows2, cols2 = matrix2.shape

    kernel_matrix = np.zeros((cols1, cols2))
    for i in range(cols1):
        for j in range(cols2):
            if kernel_class == 'rbf':
                kernel_matrix[i, j] = rbf_kernel(matrix1[:, i], matrix2[:, j])
            else:
                raise ValueError('Not this kernel')

    return kernel_matrix

if __name__ == '__main__':
    case = 'Apex'
    Y, A_true, E_true, P, N, Band = loadhsi(case)
    c = np.random.randint(0, 94)
    y = Y[c, c, :][:, np.newaxis]

    alpha = 0.5
    mu = 0.5
    lamb = 1.0
    kernel_class = 'rbf'
    x = matrix_kernel(E_true, y,kernel_class)
    w1_init, b1_init, w2_init, c2_init, b2_init, w3_init, b3_init,c3_init,  w4_init, w5_init = inithyper(E_true,  alpha, mu, lamb, kernel_class)
    print( w1_init.shape, b1_init.shape,  w2_init, c2_init, b2_init.shape, w3_init.shape, b3_init.shape, c3_init.shape, w4_init, w5_init.shape)