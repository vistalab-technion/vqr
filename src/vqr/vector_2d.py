from typing import Tuple

import cvxpy as cp
import numpy as np
from numpy import sqrt, array, quantile
from scipy.spatial.distance import cdist

EPS = 0.0001


def b_to_quantile(f, T, U1, U2, step):
    fact = 10 / step
    l = T.shape[0]
    m = U1.shape[0]
    D1 = np.zeros(m)
    D2 = np.zeros(m)
    for i1 in range(1, l):
        u1 = T[i1]
        for i2 in range(1, l):
            u2 = T[i2]
            j = np.where((fact * U1 + U2) == (fact * u1 + u2))
            jprecx = np.where(abs((fact * U1 + U2) - (fact * (u1 - step) + u2)) < EPS)
            jprecy = np.where(abs((fact * U1 + U2) - (fact * u1 + (u2 - step))) < EPS)
            D1[j] = (f[j] - f[jprecx]) / step
            D2[j] = (f[j] - f[jprecy]) / step
    return D1, D2


def vector_quantile(Y: array, U: array) -> Tuple[array, array]:
    assert Y.shape[1] == U.shape[1]  # need to have same dimensions
    N, d = Y.shape
    T = sqrt(U.shape[0])
    S = cdist(U, Y, metric=lambda x, y: x.dot(y))
    Td = int(T ** d)
    one_N = np.ones([N, 1])
    one_T = np.ones([Td, 1])
    Pi_cp = cp.Variable(shape=(Td, N))
    objective = cp.sum(cp.multiply(Pi_cp, S))
    constraints = [
        Pi_cp @ one_N == 1 / Td * one_T,
        Pi_cp >= 0,
        one_T.T @ Pi_cp == 1 / N * one_N.T,
    ]
    problem = cp.Problem(objective=cp.Maximize(objective), constraints=constraints)
    problem.solve(
        verbose=True,
    )
    u = 1 / T * (np.arange(0, T) + 1)
    U1, U2 = np.meshgrid(
        *(
            [
                u,
            ]
            * d
        )
    )
    Q1, Q2 = b_to_quantile(
        constraints[0].dual_value, u, U1.reshape(-1), U2.reshape(-1), 1 / T
    )
    return Q1, Q2


def separable_quantile(Y: array, U: array):
    T = sqrt(U.shape[0])
    alphas = 1 / T * (np.arange(0, T) + 1)
    Q1 = quantile(Y[:, 0], alphas)
    Q2 = quantile(Y[:, 1], alphas)
    return Q1, Q2
