import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve_triangular


def householder(A, kmax=None):
    """
    Given a real mxn matrix A, find the reduction to upper triangular matrix R
    using Householder transformations.

    :param A: an mxn-dimensional numpy array
    :param kmax: an integer, the number of columns of A to reduce \
    to upper triangular. If not present, will default to n.

    :return R: an mxn-dimensional numpy array containing the upper \
    triangular matrix
    """

    m, n = A.shape

    if kmax is None:
        kmax = min(m, n)

    for k in range(kmax):

        # Take the k,...,m entries of the k-th column of A
        x = A[k:, k]

        # Setup first standard basis vector of size m-k
        e = np.zeros(m-k)
        e[0] = 1

        # v = sgn(x_0)*norm(x)*e + x (with sgn(0)=1) and then normalise
        if x[0] == 0:
            v = norm(x)*e + x
        else:
            v = (x[0]/np.abs(x[0]))*norm(x)*e + x

        if norm(v) != 0:
            v = np.reshape(v / norm(v), (m-k, 1))

        # Adjust the appropriate minor of A in-place with appropriate formula
        A[k:, k:] -=  2*np.dot(v, np.dot(v.conj().T, A[k:, k:]))

    return A


def householder_solve(A, b):
    """
    Given a real mxm matrix A, use the Householder transformation to solve
    Ax_i=b_i, i=1,2,...,k.

    :param A: an mxm-dimensional numpy array
    :param b: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors b_1,b_2,...,b_k.

    :return x: an mxk-dimensional numpy array whose columns are the \
    right-hand side vectors x_1,x_2,...,x_k.
    """

    m = A.shape[0]

    # Form Ahat, the matrix of A concatenated with the b vectors
    Ahat = np.concatenate((A, b), axis=1)

    # Mulitply Ahat by Q* to transform the A part into R,
    # and the b vectors into Q*b vectors
    # Ax = b  iff  Rx = Q*b
    Ahat_house = householder(Ahat)

    # Extract R and the Q*b vectors from the transformed Ahat
    R = Ahat_house[:, :m]
    bhat = Ahat_house[:, m:]

    # Solve Rx = Q*b by back substitution
    x = solve_triangular(R, bhat)

    return x


def householder_qr(A):
    """
    Given a real mxn matrix A, use the Householder transformation to find
    the QR factorisation of A.

    :param A: an mxn-dimensional numpy array

    :return Q: an mxm-dimensional numpy array
    :return R: an mxn-dimensional numpy array
    """

    m, n = A.shape

    # Form Ahat, the matrix of A concatenated with the identity matrix of size m
    Ahat = np.concatenate((A, np.eye(m)), axis=1)

    # Mulitply Ahat by Q* to transform the A part into R,
    # and the identity into Q*
    Ahat_house = householder(Ahat)

    # Extract R and Q* from the transformed Ahat
    R = Ahat_house[:, :n]
    Q_star = Ahat_house[:, n:]

    # Get Q from Q* the unitary matrix property
    Q = Q_star.conj().T

    return Q, R


def householder_ls(A, b):
    """
    Given a real mxn matrix A and an m dimensional vector b, find the
    least squares solution to Ax = b.

    :param A: an mxn-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an n-dimensional numpy array
    """

    m, n = A.shape

    # Form Ahat, the matrix of A concatenated with b
    Ahat = np.c_[A, b]

    # Mulitply Ahat by Q* to transform the A part into R,
    # and b into Q*b
    Ahat_house = householder(Ahat)

    # Extract Rhat and Qhat*b from the transformed Ahat,
    # where Rhat and Qhat are the reduced versions of R and Q respectively
    Rhat = Ahat_house[:n, :n]
    bhat = Ahat_house[:n, n:]

    # Solve Rhatx = Q*b by back substitution
    x = solve_triangular(Rhat, bhat).reshape(n,)

    return x
