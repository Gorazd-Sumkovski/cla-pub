import numpy as np
from cla_utils.exercises4 import operator_2_norm


def randomQ(m):
    """
    Produce a random orthogonal mxm matrix.

    :param m: the matrix dimension parameter.

    :return Q: the mxm numpy array containing the orthogonal matrix.
    """

    Q, R = np.linalg.qr(np.random.randn(m, m))

    return Q


def randomR(m):
    """
    Produce a random upper triangular mxm matrix.

    :param m: the matrix dimension parameter.

    :return R: the mxm numpy array containing the upper triangular matrix.
    """

    A = np.random.randn(m, m)

    return np.triu(A)


def backward_stability_householder(m):
    """
    Verify backward stability for QR factorisation using Householder for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """

    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        Q1 = randomQ(m)
        R1 = randomR(m)

        A = Q1 @ R1
        Q2, R2 = np.linalg.qr(A)

        print(f'||Q2 - Q1|| = {operator_2_norm(Q2 - Q1)}')
        print(f'||R2 - R1|| = {operator_2_norm(R2 - R1)}')
        print(f'||A - Q2*R2|| = {operator_2_norm(A - Q2@R2)}')
        print('')


def solve_R(R, b):
    """
    Solve the system Rx=b where R is an mxm upper triangular matrix
    and b is an m dimensional vector.

    :param R: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :param x: an m-dimensional numpy array
    """

    m = R.shape[0]
    x = np.zeros(m)

    # Calculate last entry first, simply by dividing
    x[m-1] = b[m-1]/R[m-1, m-1]

    # Back sub
    for i in range(m-2, -1, -1):
        x[i] = (b[i] - np.dot(R[i, (i+1):], x[(i+1):])) / R[i, i]

    return x


def back_stab_solve_R(m):
    """
    Verify backward stability for back substitution for
    real mxm matrices.

    :param m: the matrix dimension parameter.
    """
    # repeat the experiment a few times to capture typical behaviour
    for k in range(20):
        A = np.random.randn(m, m)
        R = np.triu(A)

        x1 = np.random.randn(m)
        b1 = np.dot(R, x1)

        x2 = solve_R(R, b1)
        b2 = np.dot(R, x2)

        print(f'||x2 - x1|| = {np.linalg.norm(x2 - x1)}')
        print(f'||b2 - b1|| = {np.linalg.norm(b2 - b1)}')
        print(f'||R*x2 - b1|| = {np.linalg.norm(np.dot(R, x2) - b1)}')
        print('')


def back_stab_householder_solve(m):
    """
    Verify backward stability for the householder algorithm
    for solving Ax=b for an m dimensional square system.

    :param m: the matrix dimension parameter.
    """
    raise NotImplementedError
