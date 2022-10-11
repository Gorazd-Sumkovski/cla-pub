import numpy as np
from numpy.linalg import norm, eig
from numpy.random import randint, rand


def operator_2_norm(A):
    """
    Given a real mxn matrix A, return the operator 2-norm.

    :param A: an mxn-dimensional numpy array

    :return o2norm: the norm
    """

    w = eig(A.T @ A)[0]
    lamda = np.amax(w)
    o2norm = np.sqrt(lamda)

    return o2norm

def op_2_norm_ineq_verifier(num_tests):

    for i in range(num_tests):
        m = randint(500)
        n = randint(500)
        k = randint(1000)

        A = k*rand(m, n)
        x = k*rand(n)

        print(norm(A.dot(x)) <= operator_2_norm(A) * norm(x))


def AB_op_norm_ineq_verifier(num_tests):

    for i in range(num_tests):
        m = randint(500)
        n = randint(500)
        l = randint(500)
        k = randint(1000)

        A = k*rand(l, m)
        B = k*rand(m, n)

        print(operator_2_norm(A @ B) <= operator_2_norm(A) * operator_2_norm(B))


def cond(A):
    """
    Given a real mxn matrix A, return the condition number in the 2-norm.

    :param A: an mxn-dimensional numpy array

    :return ncond: the condition number
    """

    w = eig(A @ A.T)[0]
    eta = np.amin(w)
    op_norm_Ainv = np.sqrt(1/eta)

    ncond = op_norm_Ainv * operator_2_norm(A)

    return ncond
