from typing import Tuple

import cvxpy as cp
import numpy as np
import pylab as pl
import numpy.random
from numpy import sqrt, array, quantile
from numpy.random import shuffle
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

numpy.random.seed(42)
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


def generate_mvn(N: int = 2000, rho: float = 0.6) -> array:
    Y = np.random.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]], size=(N,))
    return Y


def generate_heart():
    # generates 2k-3k points heart
    image = 1 - pl.imread("../../data/heart.png")[:, :, 2]
    image = image[::2, ::2]
    image = image / np.sum(image)
    idces = image.nonzero()
    Y = np.zeros([len(idces[0]), 2])
    Y[:, 0] = idces[0] / idces[0].max()
    Y[:, 1] = idces[1] / idces[1].max()
    return Y


def generate_star():
    image = 1 - pl.imread("../../data/star.jpg") / 255
    image = image[::7, ::7]
    image = image / np.sum(image)
    idces = image.nonzero()
    Y = np.zeros([len(idces[0]), 2])
    Y[:, 0] = idces[0] / idces[0].max()
    Y[:, 1] = idces[1] / idces[1].max()
    return Y


def split_calibration_test(data: array, split_ratio: float) -> Tuple[array, array]:
    N, _ = data.shape
    idxes = np.arange(0, N)
    shuffle(idxes)
    calibration_data = data[idxes[0 : int(N * split_ratio)], :]
    test_data = data[
        idxes[int(N * split_ratio) :],
    ]
    return calibration_data, test_data


def measure_coverage(quantile_surface: array, data: array):
    def point_in_hull(point, hull, tolerance=1e-12):
        return all(
            (np.dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations
        )

    cvx_hull = ConvexHull(quantile_surface)
    coverage = [point_in_hull(d, cvx_hull) for d in data]  # type: ignore
    return (sum(coverage) / len(coverage)) * 100


def run_vector_quantile_experiment(
    distribution: str = "mvn",
    N: int = 1000,
    split_ratio: float = 0.5,
    quantile_type: str = "vector",
):
    d = 2
    T = 40  # then [1, -2] should have 90% coverage
    u = 1 / T * (np.arange(0, T) + 1)
    U1, U2 = np.meshgrid(
        *(
            [
                u,
            ]
            * d
        )
    )  # not handling d>2
    U = np.stack([U1.reshape(-1), U2.reshape(-1)], axis=1)
    if distribution == "mvn":
        Y = generate_mvn(N, rho=0.6)
    elif distribution == "heart":
        Y = generate_heart()
    elif distribution == "star":
        Y = generate_star()
    else:
        NotImplementedError(
            f"Unrecognized distribution: {distribution}. "
            f"Only the following are supported: star, heart, mvn."
        )

    Y_calib, Y_test = split_calibration_test(Y, split_ratio)

    if quantile_type == "vector":
        # fit quantile curves on calibration
        Q1, Q2 = vector_quantile(Y_calib, U)
        Q1 = Q1.reshape([T, T])
        Q2 = Q2.reshape([T, T])

        # construct a quantile surface
        # when t = 40, 1:-2 is equivalent to asking 0.05 - 0.95
        # thus we expect 90% coverage
        surface = np.array(
            [
                [*Q1[1:-2, -2], *Q1[1, 1:-2], *Q1[1:-2, 1], *Q1[-2, 1:-2]],
                [*Q2[1:-2, -2], *Q2[1, 1:-2], *Q2[1:-2, 1], *Q2[-2, 1:-2]],
            ]
        ).T
    elif quantile_type == "separable":
        Q1, Q2 = separable_quantile(Y, U)
        surface = np.array(
            [
                [
                    *Q1[1:-2],
                    *Q1[1:-2],
                    *[Q1[1]] * len(Q2[1:-2]),
                    *[Q1[-2]] * len(Q2[1:-2]),
                ],
                [
                    *[Q2[-2]] * len(Q1[1:-2]),
                    *[Q2[1]] * len(Q1[1:-2]),
                    *Q2[1:-2],
                    *Q2[1:-2],
                ],
            ]
        ).T
    else:
        NotImplementedError(
            f"Unrecognized {quantile_type=}. " f"Supporting only: vector, separable."
        )

    return measure_coverage(surface, Y_test)


if __name__ == "__main__":
    num_trials = 10
    coverages = []
    for _ in range(num_trials):
        coverage = run_vector_quantile_experiment(
            distribution="heart", N=2000, quantile_type="separable"
        )
        print(coverage)
        coverages.append(coverage)
    print(
        f"Marginal coverage over {num_trials} trials: "
        f"{round(sum(coverages)/len(coverages), 3)}%"
    )
