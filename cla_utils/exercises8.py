import numpy as np
from numpy.linalg import norm

def Q1AQ1s(A):
    """
    For a matrix A, find the unitary matrix Q1 such that the first
    column of Q1*A has zeros below the diagonal. Then return A1 = Q1*A*Q1^*.

    :param A: an mxm numpy array

    :return A1: an mxm numpy array
    """

    m = A.shape[0]

    # Just like the first step of Householder
    x = A[:, 0]
    e = np.zeros(m)
    e[0] = 1

    if x[0] == 0:
        v = norm(x)*e + x
    else:
        v = np.sign(x[0])*norm(x)*e + x

    v = np.reshape(v / norm(v), (m, 1))

    # Left multiplying A by Q1
    A -= 2*np.dot(v, np.dot(v.conj().T, A))
    # Right multiplying Q1A by Q1*
    A -= 2*np.dot(np.dot(A, v), v.conj().T)

    return A


def hessenberg(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place.

    :param A: an mxm numpy array
    """

    m = A.shape[0]

    for k in range(m-2):
        x = A[(k+1):, k]
        e = np.zeros(m-k-1)
        e[0] = 1

        if x[0] == 0:
            v = norm(x)*e + x
        else:
            v = np.sign(x[0])*norm(x)*e + x

        v = np.reshape(v / norm(v), (m-k-1, 1))

        A[(k+1):, k:] -= 2*np.dot(v, np.dot(v.conj().T, A[(k+1):, k:]))
        A[:, (k+1):] -= 2*np.dot(np.dot(A[:, (k+1):], v), v.conj().T)

    return A


def hessenbergQ(A):
    """
    For a matrix A, transform to Hessenberg form H by Householder
    similarity transformations, in place, and return the matrix Q
    for which QHQ^* = A.

    :param A: an mxm numpy array

    :return Q: an mxm numpy array
    """

    m = A.shape[0]
    Q = np.eye(m)

    for k in range(m-2):
        x = A[(k+1):, k]
        e = np.zeros(m-k-1)
        e[0] = 1

        if x[0] == 0:
            v = norm(x)*e + x
        else:
            v = np.sign(x[0])*norm(x)*e + x

        v = np.reshape(v / norm(v), (m-k-1, 1))

        A[(k+1):, k:] -= 2*np.dot(v, np.dot(v.conj().T, A[(k+1):, k:]))
        Q[(k+1):, :] -= 2*np.dot(v, np.dot(v.conj().T, Q[(k+1):, :]))
        A[:, (k+1):] -= 2*np.dot(np.dot(A[:, (k+1):], v), v.conj().T)

    return Q.conj().T


def hessenberg_ev(H):
    """
    Given a Hessenberg matrix, return the eigenvalues and eigenvectors.

    :param H: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of H
    """

    m, n = H.shape
    assert(m == n)
    assert(np.linalg.norm(H[np.tril_indices(m, -2)]) < 1.0e-6)
    _, V = np.linalg.eig(H)
    return V


def ev(A):
    """
    Given a matrix A, return its eigenvectors. This should
    be done by using your functions to reduce to upper Hessenberg
    form, before calling hessenberg_ev (which you should not edit!).

    :param A: an mxm numpy array

    :return V: an mxm numpy array whose columns are the eigenvectors of A
    """

    Q = hessenbergQ(A)
    V = hessenberg_ev(A)

    return Q@V
