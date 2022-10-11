import numpy as np
import numpy.random as random
from cla_utils.exercises3 import householder_qr

def get_A100():
    """
    Return A100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    return A


def get_B100():
    """
    Return B100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A[np.tril_indices(m, -2)] = 0
    return A


def get_C100():
    """
    Return C100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    return A


def get_D100():
    """
    Return D100 matrix for investigating QR factoration.

    :return A: The 100x100 numpy array
    """
    m = 100
    random.seed(1111*m)
    A = random.randn(m, m) + 1j*random.randn(m, m)
    A = 0.5*(A + np.conj(A).T)
    A[np.tril_indices(m, -2)] = 0
    A[np.triu_indices(m, 2)] = 0
    return A


def get_A3():
    """
    Return A3 matrix for investigating power iteration.

    :return A3: a 3x3 numpy array.
    """

    return array([[ 0.68557183+0.46550108j,  0.12934765-0.1622676j ,
                    0.24409518+0.25335939j],
                  [ 0.1531015 +0.66678983j,  0.45112492+0.18206976j,
                    -0.02633966+0.43477693j],
                  [-0.10817164-1.16879196j, -0.18446849+0.03755672j,
                   0.06430325-0.44757084j]])


def get_B3():
    """
    Return B3 matrix for investigating power iteration.

    :return B3: a 3x3 numpy array.
    """
    return array([[ 0.46870499+0.37541453j,  0.19115959-0.39233203j,
                    0.12830659+0.12102382j],
                  [ 0.90249603-0.09446345j,  0.51584055+0.84326503j,
                    -0.02582305+0.23259079j],
                  [ 0.75419973-0.52470311j, -0.59173739+0.48075322j,
                    0.51545446-0.21867957j]])


def pow_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a matrix A, apply the power iteration algorithm with initial
    guess x0, until either

    ||r|| < tol where

    r = Ax - lambda*x,

    or the number of iterations exceeds maxit.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of power iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing all \
    the iterates.
    :return lambda0: the final eigenvalue.
    """

    m = A.shape[0]

    # To store each x iterate for when store_iterations = True
    x_store = np.zeros((m, maxit))

    # Number of iterations (algo breaks if k reaches maxit)
    k = 0

    # Initial eigval guess, if this makes the error be within tolerance
    # then we are done, we have the eigval and eigvec
    lambda0 = np.dot(x0.conj(), A@x0)

    if store_iterations == True:
        # Iterate until maxit iterations or
        # until error is within tolerance
        while k < maxit and np.linalg.norm(A@x0 - lambda0*x0) > tol:
            # Power iteration
            w = A@x0
            # Normalise next iterate
            x0 = w/np.linalg.norm(w)
            # By the above normalisation the Rayleigh quot is just
            lambda0 = np.dot(x0,conj(), A@x0)

            # Update and store latest iterate
            x_store[:, k] = x0
            k += 1

        return x_store, lambda0

    # If store_iterations = False then same thing but wihout storing
    else:
        while k < maxit and np.linalg.norm(A@x0 - lambda0*x0) > tol:
            # Power iteration
            w = A@x0
            # Normalise next iterate
            x0 = w/np.linalg.norm(w)
            # By the above normalisation the Rayleigh quot is just
            lambda0 = np.dot(x0.conj(), A@x0)

            # Update
            k += 1

        return x0, lambda0


def inverse_it(A, x0, mu, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the inverse iteration algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param mu: a floating point number, the shift parameter
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """

    m = A.shape[0]
    I = np.eye(m)

    # To store each of the iterates for when store_iterations = True
    x_store = np.zeros((m, maxit))
    l_store = np.zeros(maxit)

    # Number of iterations (algo breaks if k reaches maxit)
    k = 0

    # Initial eigval guess, if this makes the error be within tolerance
    # then we are done, we have the eigval and eigvec
    l0 = np.dot(x0.conj(), A@x0)

    if store_iterations == True:
        while k < maxit and np.linalg.norm(A@x0 - l0*x0) > tol:
            # Power iteration
            w = np.linalg.solve(A - mu*I, x0)
            # Normalise next iterate
            x0 = w/np.linalg.norm(w)
            # By the above normalisation the Rayleigh quot is just
            l0 = np.dot(x0.conj(), A@x0)

            # Update and store latest iterates
            x_store[:, k] = x0
            l_store[k] = l0
            k += 1

        return x_store, l_store

    # If store_iterations = False then same thing but wihout storing
    else:
        while k < maxit and np.linalg.norm(A@x0 - l0*x0) > tol:
            # Power iteration
            w = np.linalg.solve(A - mu*I, x0)
            # Normalise next iterate
            x0 = w/np.linalg.norm(w)
            # By the above normalisation the Rayleigh quot is just
            l0 = np.dot(x0.conj(), A@x0)

            # Update
            k += 1

        return x0, l0


def rq_it(A, x0, tol, maxit, store_iterations = False):
    """
    For a Hermitian matrix A, apply the Rayleigh quotient algorithm
    with initial guess x0, using the same termination criteria as
    for pow_it.

    :param A: an mxm numpy array
    :param x0: the starting vector for the power iteration
    :param tol: a positive float, the tolerance
    :param maxit: integer, max number of iterations
    :param store_iterations: if True, then return the entire sequence \
    of inverse iterates, instead of just the final iteration. Default is \
    False.

    :return x: an m dimensional numpy array containing the final iterate, or \
    if store_iterations, an mxmaxit dimensional numpy array containing \
    all the iterates.
    :return l: a floating point number containing the final eigenvalue \
    estimate, or if store_iterations, an m dimensional numpy array containing \
    all the iterates.
    """

    m = A.shape[0]
    I = np.eye(m)
    # lambda0 as defined
    l0 = np.dot(x0.conj(), A@x0)

    # To store each of the iterates for when store_iterations = True
    x_store = np.zeros((m, maxit))
    l_store = np.zeros(maxit)

    # Number of iterations (algo breaks if k reaches maxit)
    k = 0

    # Initial eigval guess, if this makes the error be within tolerance
    # then we are done, we have the eigval and eigvec
    l0 = np.dot(x0.conj(), A@x0)

    if store_iterations == True:
        while k < maxit and np.linalg.norm(A@x0 - l0*x0) > tol:
            # Power iteration
            w = np.linalg.solve(A - l0*I, x0)
            # Normalise next iterate
            x0 = w/np.linalg.norm(w)
            # By the above normalisation the Rayleigh quot is just
            l0 = np.dot(x0.conj(), A@x0)

            # Update and store latest iterates
            x_store[:, k] = x0
            l_store[k] = l0
            k += 1

        return x_store, l_store

    # If store_iterations = False then same thing but wihout storing
    else:
        while k < maxit and np.linalg.norm(A@x0 - l0*x0) > tol:
            # Power iteration
            w = np.linalg.solve(A - l0*I, x0)
            # Normalise next iterate
            x0 = w/np.linalg.norm(w)
            # By the above normalisation the Rayleigh quot is just
            l0 = np.dot(x0.conj(), A@x0)

            # Update
            k += 1

        return x0, l0


def pure_QR(A, maxit, tol, return_errors=False):
    """
    For matrix A, apply the QR algorithm and return the result.

    :param A: an mxm numpy array
    :param maxit: the maximum number of iterations
    :param tol: termination tolerance
    :param return_errors: logical, if True returns array of errors

    :return Ak: the result
    """

    # To store errors at each iteration
    if return_errors:
        arr = np.array([])

    # Number of iterations (algo breaks if k reaches maxit)
    k = 0
    # Error, needs to initialised to something > tol
    err = tol + 1

    while k < maxit and err > tol:
        # QR fac A
        Q, R = householder_qr(A)
        # Reverse Q and R as outlined in the algo
        A = R@Q

        # A should be upper tri (and diag in symmetric case)
        err = np.linalg.norm(np.tril(A, -1))
        # Update
        k += 1

        # Store error
        if return_errors:
            arr = np.append(arr, err)

    if return_errors:
        return A, arr
    else:
        return A
