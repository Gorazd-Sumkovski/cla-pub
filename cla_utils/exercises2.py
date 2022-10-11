import numpy as np
import timeit
import numpy.random as random
from numpy.linalg import solve, qr, norm

# pre-constructions in the namespace to use in tests
A0 = random.randn(100, 100)
Q0 = qr(A0)[0]
b0 = random.randn(100)

A1 = random.randn(200, 200)
Q1 = qr(A1)[0]
b1 = random.randn(200)

A2 = random.randn(400, 400)
Q2 = qr(A2)[0]
b2 = random.randn(400)

def orthog_cpts(v, Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute v = r + u_1q_1 + u_2q_2 + ... + u_nq_n
    for scalar coefficients u_1, u_2, ..., u_n and
    residual vector r

    :param v: an m-dimensional numpy array
    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return r: an m-dimensional numpy array containing the residual
    :return u: an n-dimensional numpy array containing the coefficients
    """

    n = Q.shape[1]
    u = np.zeros(n)

    for i in range(n):
        u[i] = Q[:,i].conj().dot(v)

    r = v - Q.dot(u)

    return r, u


def solveQ(Q, b):
    """
    Given a unitary mxm matrix Q and a vector b, solve Qx=b for x.

    :param Q: an mxm dimensional numpy array containing the unitary matrix
    :param b: the m dimensional array for the RHS

    :return x: m dimensional array containing the solution.
    """

    x = Q.conj().T.dot(b)

    return x


def timetable_solveQ_100():

    """
    Doing a solveQ example with an input matrix of size 100 to pass to timeit.
    """

    x = solveQ(Q0,b0)


def timetable_numpy_solve_100():

    """
    Doing a numpy example with an input matrix of size 100 to pass to timeit.
    """

    x = solve(Q0,b0)


def timetable_solveQ_200():

    """
    Doing a solveQ example with an input matrix of size 200 to pass to timeit.
    """

    x = solveQ(Q1,b1)


def timetable_numpy_solve_200():

    """
    Doing a numpy example with an input matrix of size 200 to pass to timeit.
    """

    x = solve(Q1,b1)


def timetable_solveQ_400():

    """
    Doing a solveQ example with an input matrix of size 400 to pass to timeit.
    """

    x = solveQ(Q2,b2)


def timetable_numpy_solve_400():

    """
    Doing a numpy example with an input matrix of size 400 to pass to timeit.
    """

    x = solve(Q2,b2)


def time_solve():

    """
    Get some timings for solves.
    """

    print("Timing for solveQ with input matrix of size 100")
    print(timeit.Timer(timetable_solveQ_100).timeit(number=1))
    print("Timing for numpy with input matrix of size 100")
    print(timeit.Timer(timetable_numpy_solve_100).timeit(number=1))
    print()
    print("Timing for solveQ with input matrix of size 200")
    print(timeit.Timer(timetable_solveQ_200).timeit(number=1))
    print("Timing for numpy with input matrix of size 200")
    print(timeit.Timer(timetable_numpy_solve_200).timeit(number=1))
    print()
    print("Timing for solveQ with input matrix of size 400")
    print(timeit.Timer(timetable_solveQ_400).timeit(number=1))
    print("Timing for numpy with input matrix of size 400")
    print(timeit.Timer(timetable_numpy_solve_400).timeit(number=1))


def orthog_proj(Q):
    """
    Given a vector v and an orthonormal set of vectors q_1,...q_n,
    compute the orthogonal projector P that projects vectors onto
    the subspace spanned by those vectors.

    :param Q: an mxn-dimensional numpy array whose columns are the \
    orthonormal vectors

    :return P: an mxm-dimensional numpy array containing the projector
    """

    m, n = Q.shape
    P = np.zeros((m,m), dtype='complex128')

    for i in range(n):
        P += np.outer(Q[:,i] , Q[:,i].conj())

    return P


def orthog_space(V):
    """
    Given set of vectors u_1,u_2,..., u_n, compute the
    orthogonal complement to the subspace U spanned by the vectors.

    :param V: an mxn-dimensional numpy array whose columns are the \
    vectors u_1,u_2,...,u_n.

    :return Q: an lxm-dimensional numpy array whose columns are an \
    orthonormal basis for the subspace orthogonal to U.
    """

    n = V.shape[1]

    Q, R = qr(V, mode='complete')
    Q = Q[:, n:]

    return Q


def GS_classical(A):
    """
    Given an mxn matrix A, compute the QR factorisation by classical
    Gram-Schmidt algorithm.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    A = 1.0*A
    m, n = A.shape
    Q = np.zeros((m,n), dtype = A.dtype)
    R = np.zeros((n,n), dtype = A.dtype)

    R[0, 0] = norm(A[:, 0])
    Q[:, 0] = A[:, 0] / R[0, 0]

    for j in range(1, n):
        R[:j-1, j] = np.dot(Q[:, :j-1].conj().T, A[:, j])
        A[:, j] -= np.dot(Q[:, :j-1], R[:j-1, j])

        R[j, j] = norm(A[:, j])
        Q[:, j] = A[:, j] / R[j, j]

    return Q, R


def GS_modified(A):
    """
    Given an mxn matrix A, compute the QR factorisation by modified
    Gram-Schmidt algorithm, producing

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    A = A.copy()
    m, n = A.shape
    Q = np.zeros((m,n), dtype = A.dtype)
    R = np.zeros((n,n), dtype = A.dtype)

    for i in range(n):
        R[i, i] = norm(A[:, i])
        Q[:, i] = A[:, i] / R[i, i]

        R[i, i+1:] = np.dot(A[:, i+1:].T, Q[:, i].conj())
        A[:, i+1:] -= np.outer(Q[:, i], R[i, i+1:])

    return Q, R


def GS_modified_get_R(A, k):
    """
    Given an mxn matrix A, with columns of A[:, 0:k] assumed orthonormal,
    return upper triangular nxn matrix R such that
    Ahat = A*R has the properties that
    1) Ahat[:, 0:k] = A[:, 0:k],
    2) A[:, k] is orthogonal to the columns of A[:, 0:k].

    :param A: mxn numpy array
    :param k: integer indicating the column that R should orthogonalise

    :return R: nxn numpy array
    """

    n = A.shape[1]
    R = np.eye(n)

    R[k, k] = norm(A[:, k])

    for i in range(1, n-k):
        R[k, k+i] = R[k, k] * np.inner(A[:, k], A[:, k+i])

    return R


def GS_modified_R(A):
    """
    Implement the modified Gram Schmidt algorithm using the lower triangular
    formulation with Rs provided from GS_modified_get_R.

    :param A: mxn numpy array

    :return Q: mxn numpy array
    :return R: nxn numpy array
    """

    m, n = A.shape
    A = 1.0*A
    R = np.eye(n, dtype=A.dtype)
    for i in range(n):
        Rk = GS_modified_get_R(A, i)
        A[:,:] = np.dot(A, Rk)
        R[:,:] = np.dot(R, Rk)
    R = np.linalg.inv(R)
    return A, R
