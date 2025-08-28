from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt


def SAD(E, E_hat, num_endmembers, endmembers_name=None): #sad
    '''
    :param E: c x P
    :param E_hat: c x P
    :param num_endmembers: P
    :return: 
    '''
    if endmembers_name is None:
        endmembers_name = ['angle of {}'.format(i + 1) for i in range(num_endmembers)]
    angle = E * E_hat

    E_normed = LA.norm(E, axis=0)
    E_hat_normed = LA.norm(E_hat, axis=0)

    su = np.sum((angle / (E_normed * E_hat_normed).T), axis=0)
    cos = np.clip(su, -1, 1)
    sad = np.arccos(cos) * (180 / np.pi)
    sad_ = dict(zip(endmembers_name, sad))
    print(sad_)
    all_sad = np.sum(sad)
    print('Overall SAD:{}'.format(all_sad / num_endmembers))
    return all_sad / num_endmembers



def RMSE(A, A_hat, num_endmembers, endmembers_name=None):
    '''
    :param A: P x h x w
    :param A_hat: P x h x w
    :param num_endmembers: P
    :return: 
    '''
    if endmembers_name is None:
        endmembers_name = ['angle of {}'.format(i + 1) for i in range(num_endmembers)]
    diff = A - A_hat
    all_rmse = 100 * np.sqrt(np.mean(diff ** 2, axis=(1, 2)))
    rmse_ = dict(zip(endmembers_name, all_rmse))
    rmse = 100 * np.sqrt(((A - A_hat) ** 2).mean())
    print(rmse_)
    print('Overall RMSE: {}'.format(rmse))
    return rmse

def draw_E(E_true, E_test, P, norm=True):
    if norm is True:
        E_true = E_true / np.max(E_true, axis=0)
        E_test = E_test / np.max(E_test, axis=0)
    for i in range(P):
        plt.subplot(3, 2, i+1)
        plt.plot(E_test[:, i])
        plt.plot(E_true[:, i])
    plt.show()


def draw_A(A_true, A_test, P):
    for i in range(P):
        plt.subplot(2, P, i + 1)
        plt.imshow(A_test[i, :, :],  cmap='jet', interpolation='none')
    for i in range(P):
        plt.subplot(2, P, P + i + 1 )
        plt.imshow(A_true[i, :, :] , cmap='jet', interpolation='none')
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    fake1 = np.random.rand(156, 3)

    toy1 = fake1[:, 0] + 0.001 * np.random.randn(156)
    toy2 = -fake1[:, 1]
    toy3 = 2 * fake1[:, 2]
    toys1 = [toy1, toy2, toy3]
    toys1 = np.array(toys1).transpose(1, 0)

    fake2 = np.random.rand(3, 95, 95)

    toy1 = fake2[0, :] + 0.001 * np.random.randn(95, 95)
    toy2 = -fake2[1, :]
    toy3 = 2 * fake2[2, :]
    toys2 = [toy1, toy2, toy3]
    toys2 = np.array(toys2)
    print(toy1.shape, fake1.shape, toy2.shape, fake2.shape)
    SAD(fake1, toys1, 3)
    RMSE(fake2, toy2, 3)

