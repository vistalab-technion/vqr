# %%
# %load_ext autoreload
# %autoreload 2

import ot
import numpy as np

# %%
import matplotlib.pyplot as plt

# %%
from numpy import array, stack, zeros
from numpy.random import randint, uniform

from vqr.api import VectorQuantileRegressor

# %%
from vqr.vqr import decode_quantile_values

# %%
from experiments.data.mvn import LinearMVNDataProvider

d = 2
k = 2
n = 5000
X, Y_orig = LinearMVNDataProvider(d=d, k=k).sample(n=n)

T = 50

U_samples = randint(0, T, size=(n, 2))

vqr_ = VectorQuantileRegressor(
    solver_opts={"verbose": True, "num_epochs": 1000, "epsilon": 0.0001}
)
vqr_ = vqr_.fit(X, Y_orig)

A = vqr_._fitted_solution._A
B = vqr_._fitted_solution._B


# Sample X, Uniformly sample U to get Y

X_new = uniform(size=[n, k]) - 0.5
Y_samp = zeros([n, d])

for i in range(X.shape[0]):
    Y_hat = (B @ X_new[i, :][:, None]) + A
    Q1, Q2 = decode_quantile_values(T, d=2, Y_hat=Y_hat)
    u1, u2 = U_samples[i]
    Y_samp[i, :] = array((Q1[u1, u2], Q2[u1, u2]))


def U_to_colors(U_):
    U_ = np.concatenate([U_, np.zeros(shape=(U_.shape[0], 1))], axis=1)
    return U_ / T


U_colors = U_to_colors(U_samples)
X_colors = np.concatenate([X_new + 0.5, np.zeros(shape=(X_new.shape[0], 1))], axis=1)
colors = X_colors  # U_colors


# %%
def w2(Y_gt_, Y_est_):
    return ot.emd2(
        a=[], b=[], M=ot.dist(Y_gt_, Y_est_), numItermax=200_000, numThreads=8
    )


fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].scatter(Y_orig[:, 0], Y_orig[:, 1], c=colors)
ax[0].set_title("Y - original")
ax[0].set_xlim(-5, 5)
ax[0].set_ylim(-5, 5)

ax[1].scatter(Y_samp[:, 0], Y_samp[:, 1], c=colors)
ax[1].set_title(f"Y - VQR fitted and sampled (w2={w2(Y_orig, Y_samp):.4f})")
ax[1].set_xlim(-5, 5)
ax[1].set_ylim(-5, 5)

plt.show()


# %%
# Generate a quantile function that is nonlinear in X


def g(x):
    # A nonlinear function in X
    Q = array([[2.0, 1.0], [1.0, 2.0]])
    return x.T @ Q @ x + x


Y_nl = zeros([n, d])
for i in range(n):
    # Sample using a known B but the quantile function is now nonlinear in X
    Y_hat = (B @ g(X_new[i, :][:, None])) + A
    Q1, Q2 = decode_quantile_values(T, d=2, Y_hat=Y_hat)
    u1, u2 = U_samples[i]
    Y_nl[i, :] = array((Q1[u1, u2], Q2[u1, u2]))


# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].scatter(Y_samp[:, 0], Y_samp[:, 1], c=colors)
ax[0].set_title("Linear Q(Y|X)")
ax[0].set_xlim(-5, 5)
ax[0].set_ylim(-5, 5)

ax[1].scatter(Y_nl[:, 0], Y_nl[:, 1], c=colors)
ax[1].set_title(f"Nonlinear Q(Y|X) (w2={w2(Y_samp, Y_nl):.4f})")
ax[1].set_xlim(-5, 5)
ax[1].set_ylim(-5, 5)

ax[2].scatter(U_samples[:, 0] / T, U_samples[:, 1] / T, c=U_colors)
ax[2].set_title("U_samples")
ax[2].set_xlim(0, 1)
ax[2].set_xlabel("u1")
ax[2].set_ylim(0, 1)
ax[2].set_ylabel("u2")

plt.show()

# %%
# We know that the distribution on the right has a nonlinear quantile function w.r.to X.
# Now we fit a VQR on this data to get a linear approximation of this nonlinear function.
# This allows us to measure how mis-specified is the quantile function

vqr_approximating_nl = VectorQuantileRegressor(
    solver_opts={"verbose": True, "num_epochs": 1000, "epsilon": 0.001}
)
vqr_approximating_nl = vqr_approximating_nl.fit(X_new, Y_nl)
B_approximating_nl = vqr_approximating_nl._fitted_solution._B
A_approximating_nl = vqr_approximating_nl._fitted_solution._A


# %%
Y_approximated_nl = zeros([n, d])
for i in range(n):
    # Y_hat = (B_approximating_nl @ g(X_new[i, :][:, None])) + A_approximating_nl
    Y_hat = (B_approximating_nl @ X_new[i, :][:, None]) + A_approximating_nl
    Q1, Q2 = decode_quantile_values(T, d=2, Y_hat=Y_hat)
    u1, u2 = U_samples[i]
    Y_approximated_nl[i, :] = array((Q1[u1, u2], Q2[u1, u2]))

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].scatter(Y_nl[:, 0], Y_nl[:, 1], c=colors)
ax[0].set_title("Non-linear Y: GT")
ax[0].set_xlim(-5, 5)
ax[0].set_ylim(-5, 5)

ax[1].scatter(Y_approximated_nl[:, 0], Y_approximated_nl[:, 1], c=colors)
ax[1].set_title(
    f"Non-linear Y: linear approx + sampled " f"(w2={w2(Y_nl, Y_approximated_nl):.4f})"
)
ax[1].set_xlim(-5, 5)
ax[1].set_ylim(-5, 5)

plt.show()


nonlinear_vqr_gt = VectorQuantileRegressor(
    solver_opts={"verbose": True, "num_epochs": 1000, "epsilon": 0.001}
)
nonlinear_vqr_gt = nonlinear_vqr_gt.fit(stack([g(X_new[i, :]) for i in range(n)]), Y_nl)
B_nl_gt = nonlinear_vqr_gt._fitted_solution._B
A_nl_gt = nonlinear_vqr_gt._fitted_solution._A


# %%
Y_nl_gt = zeros([n, d])
for i in range(n):
    Y_hat = (B_nl_gt @ g(X_new[i, :][:, None])) + A_nl_gt
    Q1, Q2 = decode_quantile_values(T, d=2, Y_hat=Y_hat)
    u1, u2 = U_samples[i]
    Y_nl_gt[i, :] = array((Q1[u1, u2], Q2[u1, u2]))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].scatter(Y_nl[:, 0], Y_nl[:, 1], c=colors)
ax[0].set_title("Non-linear Y: GT.")
ax[0].set_xlim(-5, 5)
ax[0].set_ylim(-5, 5)

ax[1].scatter(Y_nl_gt[:, 0], Y_nl_gt[:, 1], c=colors)
ax[1].set_title(f"Non-linear Y: VQR on g(X) (w2={w2(Y_nl, Y_nl_gt):.4f})")
ax[1].set_xlim(-5, 5)
ax[1].set_ylim(-5, 5)

plt.show()

# %%
# from vqr.vqr import NonlinearRVQRDualLSESolver
# import pickle
#
# with open("nonlin-y.pkl", "wb") as f:
#     pickle.dump({'X': X_new, 'Y': Y_nl}, f)

# %%
