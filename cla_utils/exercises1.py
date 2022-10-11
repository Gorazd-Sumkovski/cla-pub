import numpy as np
import timeit
import numpy.random as random
from numpy.linalg import inv

# pre-construct a matrix in the namespace to use in tests
random.seed(1651)
A0 = random.randn(500, 500)
x0 = random.randn(500)

u0 = random.randn(400)
v0 = random.randn(400)

def basic_matvec(A, x):
    """
    Elementary matrix-vector multiplication.

    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    returns an m-dimensional numpy array which is the product of A with x

    This should be implemented using a double loop over the entries of A

    :return b: m-dimensional numpy array
    """

    #Initialise b, the m-dimensional output vector
    m = A.shape[0]
    n = A.shape[1]
    b = np.zeros(m)

    #Carry out the multiplication by calculating the entries of b
    for i in range(m):
        for j in range(n):
            b[i] += A[i,j] * x[j]

    return b


def column_matvec(A, x):
    """
    Matrix-vector multiplication using the representation of the product
    Ax as linear combinations of the columns of A, using the entries in
    x as coefficients.


    :param A: an mxn-dimensional numpy array
    :param x: an n-dimensional numpy array

    :return b: an m-dimensional numpy array which is the product of A with x

    This should be implemented using a single loop over the entries of x
    """

    #Initialise b, the m-dimensional output vector
    m = A.shape[0]
    n = A.shape[1]
    b = np.zeros(m)

    #Carry out multiplication as linear combination of columns of A
    for j in range(n):
        b += x[j] * A[:,j]

    return b


def timeable_basic_matvec():
    """
    Doing a matvec example with the basic_matvec that we can
    pass to timeit.
    """

    b = basic_matvec(A0, x0) # noqa


def timeable_column_matvec():
    """
    Doing a matvec example with the column_matvec that we can
    pass to timeit.
    """

    b = column_matvec(A0, x0) # noqa


def timeable_numpy_matvec():
    """
    Doing a matvec example with the builtin numpy matvec so that
    we can pass to timeit.
    """

    b = A0.dot(x0) # noqa


def time_matvecs():
    """
    Get some timings for matvecs.
    """

    print("Timing for basic_matvec")
    print(timeit.Timer(timeable_basic_matvec).timeit(number=1))
    print("Timing for column_matvec")
    print(timeit.Timer(timeable_column_matvec).timeit(number=1))
    print("Timing for numpy matvec")
    print(timeit.Timer(timeable_numpy_matvec).timeit(number=1))


def rank2(u1, u2, v1, v2):
    """
    Return the rank2 matrix A = u1*u2^* + u2*v2^*.

    :param u1: m-dimensional numpy array
    :param u1: m-dimensional numpy array
    :param v1: n-dimensional numpy array
    :param v2: n-dimensional numpy array
    """

    B = np.array([u1 , u2]).T
    C = np.array([v1.conj() , v2.conj()])

    A = B.dot(C)

    return A


def rank1pert_inv(u, v):
    """
    Return the inverse of the matrix A = I + uv^*, where I
    is the mxm dimensional identity matrix, with

    :param u: m-dimensional numpy array
    :param v: m-dimensional numpy array
    """

    m = len(u)
    I = np.eye(m)
    alpha = -1 / (1 + v.conj().dot(u))

    Ainv = I + alpha*np.outer(u, v.conj())

    return Ainv


def timetable_rank1pert_inv():

    """
    Doing a rank1pert_inv inv example to pass to timeit.
    """

    Ainv = rank1pert_inv(u0,v0)


def timetable_numpy_inv():

    """
    Doing a numpy inv example to pass to timeit.
    """

    Ainv = inv(np.eye(len(u0)) + np.outer(u0, v0.conj()))


def time_invs():

    """
    Get some timings for invs.
    """

    print("Timing for rank1pert_inv")
    print(timeit.Timer(timetable_rank1pert_inv).timeit(number=1))
    print("Timing for numpy inv")
    print(timeit.Timer(timetable_numpy_inv).timeit(number=1))


def ABiC(Ahat, xr, xi):
    """Return the real and imaginary parts of z = A*x, where A = B + iC
    with

    :param Ahat: an mxm-dimensional numpy array with Ahat[i,j] = B[i,j] \
    for i<=j and Ahat[i,j] = C[i,j] for i>j.

    :return zr: m-dimensional numpy arrays containing the real part of z.
    :return zi: m-dimensional numpy arrays containing the imaginary part of z.
    """

    m = Ahat.shape[0]
    zr = np.zeros(m)
    zi = np.zeros(m)

    #Splitting the column space matrix-vector multiplication summation formula
    #into real and imaginary parts we obtain a summation formula for zr and zi
    #zr = sum over j of xr_j*b_j - xi_j*c_j and
    #zi = sum over j of xr_j*c_j + xi_j*b_j

    for j in range(m):

        #Initialise j-th column of B
        b = np.zeros(m)
        #Initialise j-th column of C
        c = np.zeros(m)

        #Fetch j-th column of B from Ahat and B=B^T
        b[:j] = Ahat[j, :j]
        b[j:] = Ahat[j:, j]

        #Fetch the j-th column of C from Ahat and C=-C^T
        c[:j] = Ahat[:j, j]
        c[j+1:] = -Ahat[j, j+1:]

        #Implement the above summation formula for zr
        zr += xr[j]*b - xi[j]*c

        #Implement the above summation formula for zi
        zi += xr[j]*c + xi[j]*b

    return zr, zi
