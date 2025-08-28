import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def loadhsi(case):
    Y = None
    E_true = None
    A_true = None

    if case == 'Apex':
        file = 'dataset/Apex/Apex.mat'
        data = sio.loadmat(file)
        Y = data['Y'].T

        Y = Y.reshape(110, 110, 285)
        Bands = Y.shape[2]
        A_true = data['A']
        E_true = data['M']
        A_true = A_true.reshape(4, 110, 110)


    elif case == 'samson':
        file = 'dataset/Samson/Samson.mat'
        data = sio.loadmat(file)

        Y = data['Y']
        Y = np.reshape(Y, [156, 95, 95])

        for i, y in enumerate(Y):
            Y[i] = y.T
        Y = np.reshape(Y, [156, 9025])

        Y = Y.T.reshape((95, 95, 156))
        Bands = Y.shape[2]
        GT_file1 = 'dataset/Samson/Samson_A.mat'
        GT_file2 = 'dataset/Samson/Samson_E.mat'
        A_true = sio.loadmat(GT_file1)['A']
        E_true = sio.loadmat(GT_file2)['E']


    else:
        raise ValueError('None dataset')

    Bands = Bands
    Y = Y.astype(np.float32)
    N = Y.shape[0] * Y.shape[1]

    if A_true is not None:
        A_true = A_true.astype(np.float32)
        P = A_true.shape[0]
    if E_true is not None:
        E_true = E_true.astype(np.float32)
        P = E_true.shape[1]
    return Y, A_true, E_true, P, N, Bands


if __name__ == '__main__':
    PLOT = True
    cases = ['samson', 'Apex']
    for i in range(len(cases)):
        case = cases[i]
        Y, A_true, E_true, P, N, Bands = loadhsi(case)

        print('name:{}'.format(case))
        if E_true is None:
            print('Not E_true')
        print('Bands:{}'.format(Bands))
        print('endnembers:{}'.format(P))
        print('pixes nembers:{}\n'.format(N))

        if E_true is not None and PLOT is True:
            plt.plot(E_true)
            plt.show()

        if A_true is not None and PLOT is True:
            for i in range(P):
                plt.subplot(1, P, i + 1)
                plt.imshow(A_true[i, :, :], cmap='jet')
                plt.xticks([])
                plt.yticks([])
            plt.show()
