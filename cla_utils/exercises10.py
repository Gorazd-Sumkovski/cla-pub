import numpy as np
import numpy.random as random
from cla_utils.exercises3 import householder_ls
from cla_utils.exercises5 import solve_R


def arnoldi(A, b, k):
    """
    For a matrix A, apply k iterations of the Arnoldi algorithm,
    using b as the first basis vector.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array, the starting vector
    :param k: integer, the number of iterations

    :return Q: an mx(k+1) dimensional numpy array containing the orthonormal basis
    :return H: a (k+1)xk dimensional numpy array containing the upper \
    Hessenberg matrix
    """

    m = A.shape[0]
    Q = np.zeros((m, k+1), dtype=A.dtype)
    H = np.zeros((k+1, k), dtype=A.dtype)

    # Starting vector
    Q[:, 0] = b/np.linalg.norm(b)

    for n in range(k):
        v = np.dot(A, Q[:, n])

        H[:(n+1), n] = np.dot(Q[:, :(n+1)].conj().T, v)
        v -= np.dot(Q[:, :(n+1)], H[:(n+1), n])

        H[n+1, n] = np.linalg.norm(v)
        Q[:, n+1] = v/H[n+1, n]

    return Q, H


def GMRES(A, b, maxit, tol, x0=None, return_residual_norms=False,
          return_residuals=False, apply_pc=False):
    """
    For a matrix A, solve Ax=b using the basic GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param x0: the initial guess (if not present, use b)
    :param return_residual_norms: logical
    :param return_residuals: logical
    :param apply_pc: function solving Mx=b for a preconditioner M

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual \
    at iteration k
    """

    m = A.shape[0]

    Q = np.zeros((m, maxit+1), dtype=A.dtype)
    H = np.zeros((maxit+1, maxit), dtype=A.dtype)

    # Number of iterations
    nits = 0

    # To store norms of residuals if return_residual_norms = True
    if return_residual_norms:
        rnorms = np.array([])
    # To store residuals if return_residuals = True
    if return_residuals:
        r = np.array([])

    # cos and sin values for each Givens rotation
    c = np.zeros(maxit)
    s = np.zeros(maxit)

    # Make a deepcopy of b for when there is preconditioning
    b_copy = 1.0*b

    if apply_pc:
        # \tilde{b} = M^{-1} b
        # We just remap b to be \tilde{b}
        b = apply_pc(b)

    # e = ||b||e1
    e = np.zeros(maxit+1)
    e[0] = np.linalg.norm(b)

    if x0 is None:
        x0 = b

    #Initial guess
    Q[:, 0] = x0/np.linalg.norm(x0)

    for n in range(maxit):
        # Apply step n of Arnoldi algo
        v = np.dot(A, Q[:, n])
        if apply_pc:
            v = apply_pc(v)

        H[:(n+1), n] = np.dot(Q[:, :(n+1)].conj().T, v)
        v -= np.dot(Q[:, :(n+1)], H[:(n+1), n])

        H[n+1, n] = np.linalg.norm(v)
        Q[:, n+1] = v/H[n+1, n]

        # Apply Givens rotations to n-th col
        # We want to eliminate entry just below diag
        # Careful to use correct cos and sin vals for each row pair
        for i in range(n):
            temp = c[i]*H[i, n] + s[i]*H[i+1, n]
            H[i+1, n] = -s[i]*H[i, n] + c[i]*H[i+1, n]
            H[i, n] = temp

        # Get cos and sin vals for n-th and (n+1)-st row pair
        v1 = H[n, n]
        v2 = H[n+1, n]
        t = np.sqrt(v1**2 + v2**2)
        c[n] = v1/t
        s[n] = v2/t

        # Givens on n-th and (n+1)-st rows
        temp = c[n]*H[n, n] + s[n]*H[n+1, n]
        H[n+1, n] = -s[n]*H[n, n] + c[n]*H[n+1, n]
        H[n, n] = temp
        # e[n+1] = 0 at this stage so ignore it
        e[n+1] = -s[n]*e[n]
        e[n] = c[n]*e[n]

        # Find y that minimises ||\tilde{H}_n y - ||b||e1||,
        # having just reduced to upper tri with Givens
        y = solve_R(H[:(n+1), :(n+1)], e[:(n+1)])
        # x = \hat{Q}_n y
        x = np.dot(Q[:, :(n+1)], y)

        # Updates
        # Number of iterations done
        nits += 1
        # Current residual
        if return_residuals:
            r = np.append(r, np.dot(A, x) - b_copy)
        # Norm of residual
        # No need to recalculate norm of residual if we already have it
        if return_residual_norms:
            if return_residuals:
                # If we have the residual calculated then no need to recalculate it
                rnorms = np.append(rnorms, np.linalg.norm(r[n]))
                # Otherwise we do need to calculate the residual to get its norm
            else:
                rnorms = np.append(rnorms, np.linalg.norm(np.dot(A, x) - b_copy))

            # If current solution is within tolerance then return
            if rnorms[n] < tol:
                # Return what is required
                if return_residuals:
                    return x, nits, rnorms, r
                else:
                    return x, nits, rnorms

        # If we don't have residual norm the we need to calculate it
        elif np.linalg.norm(np.dot(A, x) - b_copy) < tol:
            # Return what is required
            if return_residuals:
                return x, nits, r
            else:
                return x, nits

    # If not converged return after maxit iterations
    if return_residual_norms and return_residuals:
        return x, -1, rnorms, r
    elif return_residual_norms:
        return x, -1, rnorms
    elif return_residuals:
        return x, -1, r
    else:
        return x, -1



def get_AA100():
    """
    Get the AA100 matrix.

    :return A: a 100x100 numpy array used in exercises 10.
    """
    AA100 = np.fromfile('AA100.dat', sep=' ')
    AA100 = AA100.reshape((100, 100))
    return AA100


def get_BB100():
    """
    Get the BB100 matrix.

    :return B: a 100x100 numpy array used in exercises 10.
    """
    BB100 = np.fromfile('BB100.dat', sep=' ')
    BB100 = BB100.reshape((100, 100))
    return BB100


def get_CC100():
    """
    Get the CC100 matrix.

    :return C: a 100x100 numpy array used in exercises 10.
    """
    CC100 = np.fromfile('CC100.dat', sep=' ')
    CC100 = CC100.reshape((100, 100))
    return CC100
