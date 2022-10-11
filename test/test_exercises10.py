'''Tests for the eighth exercise set.'''
import pytest
import cla_utils
from numpy import random
import numpy as np


@pytest.mark.parametrize('m, k', [(20, 4), (40, 20), (70, 13)])
def test_arnoldi(m, k):
    A = random.randn(m, m) + 1j*random.randn(m, m)
    b = random.randn(m) + 1j*random.randn(m)

    Q, H = cla_utils.arnoldi(A, b, k)
    assert(Q.shape == (m, k+1))
    assert(H.shape == (k+1, k))
    assert(np.linalg.norm((Q.conj().T)@Q - np.eye(k+1)) < 1.0e-6)
    assert(np.linalg.norm(A@Q[:, :-1] - Q@H) < 1.0e-6)


@pytest.mark.parametrize('m', [20, 204, 18])
def test_GMRES(m):
    A = random.randn(m, m)
    b = random.randn(m)

    x, _ = cla_utils.GMRES(A, b, maxit=1000, tol=1.0e-3)
    assert(np.linalg.norm(np.dot(A, x) - b) < 1.0e-3)

    # Test GMRES with preconditioning
    B = random.randn(m, m)
    c = random.randn(m)
    M = np.diag(np.diag(B))

    def diag_precond(b):
        """
        Takes in a diagonal matrix M and returns x, the solution to Mx=b.

        :param M: Diagonal mxm matrix
        :param b: m-dim vector
        """

        x = b/np.diag(M)

        return x

    y, _ = cla_utils.GMRES(B, c, maxit=1000, tol=1.0e-3, apply_pc=diag_precond)
    assert(np.linalg.norm(np.dot(B, y) - c) < 1.0e-3)


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)
