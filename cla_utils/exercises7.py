import numpy as np
from cla_utils.exercises6 import solve_L, solve_U

def perm(p, i, j):
    """
    For p representing a permutation P, i.e. Px[i] = x[p[i]],
    replace with p representing the permutation P_{i,j}P, where
    P_{i,j} exchanges rows i and j.

    :param p: an m-dimensional numpy array of integers.
    """

    p[i], p[j] = p[j], p[i]

    return p


def LUP_inplace(A):
    """
    Compute the LUP factorisation of A with partial pivoting, using the
    in-place scheme so that the strictly lower triangular components
    of the array contain the strictly lower triangular components of
    L, and the upper triangular components of the array contain the
    upper triangular components of U.

    :param A: an mxm-dimensional numpy array

    :return p: an m-dimensional integer array describing the permutation \
    i.e. (Px)[i] = x[p[i]]
    """

    m = A.shape[0]
    # Start of as identity permutation
    p = np.arange(m)

    for k in range(m-1):
        # Find largest entry of k-th column below k-th row
        max = np.argmax(np.abs(A[k:, k]))
        # So that it is the index wrt the entire k-th col of A
        max += k
        # Make appropriate swaps when necessary
        if (max > k):
            # The perm moving the largest entry for use
            p = perm(p, k, max)
            # Swap corresponding rows of U part
            temp = 1.0*A[k, k:]
            A[k, k:] = A[max, k:]
            A[max, k:] = temp
            # Swap corresponding rows of L part
            temp = 1.0*A[k, :k]
            A[k, :k] = A[max, :k]
            A[max, :k] = temp
        # L components
        A[(k+1):, k] = A[(k+1):, k]/A[k, k]
        # U components
        A[(k+1):, (k+1):] -= np.outer(A[(k+1):, k], A[k, (k+1):])

    return p


def solve_LUP(A, b):
    """
    Solve Ax=b using LUP factorisation.

    :param A: an mxm-dimensional numpy array
    :param b: an m-dimensional numpy array

    :return x: an m-dimensional numpy array
    """

    # We have PA = LU, then
    # Ax = b so PAx = Pb so LUx = Pb so Ux = solve_L(Pb) so
    # x = solve_U(solve_L(Pb))

    m = A.shape[0]
    I = np.eye(m)

    # LUP factorise A inplace (so that A contains the info of U and L)
    # Get perm matrix P (as a vector)
    p = LUP_inplace(A)
    # U is upper triangle of A (including the diagonal)
    U = np.triu(A, 0)
    # L is lower triangle of A (diagonal entries of L are all 1)
    L = np.tril(A, -1)
    # Change diag entries of L to be 1
    L += I

    # Calculate P*b using (Pb)[i] = b[p[i]]
    Pb = b[p]

    # Use above formula to solve for x
    x = solve_U(U, solve_L(L, Pb.reshape(m, 1)))

    return x.reshape(m,)


def det_LUP(A):
    """
    Find the determinant of A using LUP factorisation.

    :param A: an mxm-dimensional numpy array

    :return detA: floating point number, the determinant.
    """

    # Denote determinant by |X|
    #
    # PA = LU so |PA| = |LU| so |P||A| = |L||U|
    # P is a perm matrix so |P| = (-1)^r where r is the num of row swaps
    # Let p be the perm vec representing P, id the perm vec of the
    # identity perm and n the num of positions where p and id have
    # different values. Then r = m - n - 1.
    #
    # L is tri so |L| is the the product of the diag entries,
    # but the diag entries of L are all 1 hence |L| = 1
    #
    # We arrive at (-1)^r |A| = |U| so |A| = (-1)^r |U|
    # by multiplying both sides by (-1)^r
    #
    # But U is tri so |U| is the product of the diag entries
    # hence |A| = (-1)^r product(diag(U))

    m = A.shape[0]

    # LUP factorise A inplace (so that A contains the info of U and L)
    p = LUP_inplace(A)
    # U is upper triangle of A (including the diagonal)
    # We are only interested in the diag entries
    detU = A.diagonal().prod()

    #identity perm vec
    id = np.arange(m)
    # Use above formula for r
    r = m - np.count_nonzero(p == id) - 1

    # Use above formula for |A|
    detA = (-1)**r * detU

    return detA
