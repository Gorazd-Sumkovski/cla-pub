import numpy as np


def get_Lk(m, lvec):
    """
    Compute the lower triangular row operation mxm matrix L_k
    which has ones on the diagonal, and below diagonal entries
    in column k given by lvec (k is inferred from the size of lvec).

    :param m: integer giving the dimensions of L.
    :param lvec: a m-k-1 dimensional numpy array.

    :return Lk: an mxm dimensional numpy array.
    """

    k = m - len(lvec) - 1
    Lk = np.eye(m)
    Lk[(k+1):, k] = -lvec

    return Lk


def LU_inplace(A):
    """
    Compute the LU factorisation of A, using the in-place scheme so
    that the strictly lower triangular components of the array contain
    the strictly lower triangular components of L, and the upper
    triangular components of the array contain the upper triangular
    components of U.

    :param A: an mxm-dimensional numpy array
    """

    m = A.shape[0]

    for k in range(m-1):
        # L components
        A[(k+1):, k] = A[(k+1):, k]/A[k, k]
        # U components
        A[(k+1):, (k+1):] -= np.outer(A[(k+1):, k], A[k, (k+1):])

    return A


def solve_L(L, b):
    """
    Solve systems Lx_i=b_i for x_i with L lower triangular, i=1,2,...,k

    :param L: an mxm-dimensional numpy array, assumed lower triangular
    :param b: an mxk-dimensional numpy array, with ith column containing \
    b_i

    :return x: an mxk-dimensional numpy array, with ith column containing \
    the solution x_i
    """

    m, k = b.shape
    x = np.zeros((m, k))

    # Calculate first entries first, simply by dividing
    x[0, :] = b[0, :]/L[0, 0]

    # Forward sub
    for i in range(1, m):
        x[i, :] = (b[i, :] - L[i, :i]@x[:i, :]) / L[i, i]

    return x


def solve_U(U, b):
    """
    Solve systems Ux_i=b_i for x_i with U upper triangular, i=1,2,...,k

    :param U: an mxm-dimensional numpy array, assumed upper triangular
    :param b: an mxk-dimensional numpy array, with ith column containing \
    b_i

    :return x: an mxk-dimensional numpy array, with ith column containing \
    the solution x_i
    """

    m, k = b.shape
    x = np.zeros((m, k))

    # Calculate last entries first, simply by dividing
    x[m-1, :] = b[m-1, :]/U[m-1, m-1]

    # Back sub
    for i in range(m-2, -1, -1):
        x[i, :] = (b[i, :] - U[i, (i+1):]@x[(i+1):, :]) / U[i, i]

    return x


def inverse_LU(A):
    """
    Form the inverse of A via LU factorisation.

    :param A: an mxm-dimensional numpy array.

    :return Ainv: an mxm-dimensional numpy array.
    """

    # A*Ainv = I so L*U*Ainv = I so U*Ainv = solve_L(I) so
    # Ainv = solve_U(solve_L(I))

    m = A.shape[0]
    I = np.eye(m)

    # LU factorise A (inplace)
    LU = LU_inplace(A)
    # U is upper triangle of A (including the diagonal)
    U = np.triu(A, 0)
    # L is lower triangle of A (diagonal entries of L are all 1)
    L = np.tril(A, -1)
    # Change diag entries of L to be 1
    L += I

    # Implement above formula for Ainv
    Ainv = solve_U(U, solve_L(L, I))

    return Ainv
