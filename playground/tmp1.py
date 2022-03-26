import pickle

import ot
from numpy import array, zeros
from torch import tensor, float32
from matplotlib import pyplot as plt
from numpy.random import randint

from vqr.api import VectorQuantileRegressor
from vqr.vqr import NonlinearRVQRDualLSESolver, decode_quantile_values

with open("../notebooks/nonlin-y.pkl", "rb") as f:
    XY = pickle.load(f)
    f.close()


def w2(Y_gt_, Y_est_):
    return ot.emd2(
        a=[], b=[], M=ot.dist(Y_gt_, Y_est_), numItermax=200_000, numThreads=4
    )


X = XY["X"]
Y = XY["Y"]
U_samples = XY["U_samples"]
colors = XY["colors"]

n = 5000
d = 2
k = 2
T = 50
nonlinear_vqr_est = VectorQuantileRegressor(
    solver=NonlinearRVQRDualLSESolver(
        verbose=True, num_epochs=1500, epsilon=0.00001, learning_rate=0.9
    )
)
nonlinear_vqr_est.fit(X, Y)
Y_nl_est = zeros([n, k])
for i in range(n):
    Q1, Q2 = nonlinear_vqr_est.vector_quantiles(
        tensor(X[i, :][:, None], dtype=float32).T
    )
    u1, u2 = U_samples[i]
    Y_nl_est[i, :] = array((Q1[u1, u2], Q2[u1, u2]))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].scatter(Y[:, 0], Y[:, 1], c=colors)
ax[0].set_xlim(-5, 5)
ax[0].set_ylim(-5, 5)

ax[1].scatter(Y_nl_est[:, 0], Y_nl_est[:, 1], c=colors)
ax[1].set_xlim(-5, 5)
ax[1].set_ylim(-5, 5)
ax[1].set_title(f"{w2(Y, Y_nl_est)}")

plt.show()
