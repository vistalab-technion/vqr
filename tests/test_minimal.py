import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from vqr import VectorQuantileRegressor
from vqr.solvers.dual.regularized_lse import RegularizedDualVQRSolver


def test_vqr_minimal():
    N, d, k, T = 5000, 2, 1, 20
    N_test = N // 10
    seed = 42
    alpha = 0.05

    # Generate some data (or load from elsewhere).
    X, Y = make_regression(
        n_samples=N, n_features=k, n_targets=d, noise=0.1, random_state=seed
    )
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=N_test, shuffle=True, random_state=seed
    )

    # Create the VQR solver and regressor.
    vqr_solver = RegularizedDualVQRSolver(
        verbose=True, epsilon=1e-2, num_epochs=1000, lr=0.9
    )
    vqr = VectorQuantileRegressor(n_levels=T, solver=vqr_solver)

    # Fit the model on the data.
    vqr.fit(X_train, Y_train)

    # Conditional coverage calculation: for each test point, calculate the
    # conditional quantiles given x, and check whether the corresponding y is covered
    # in the alpha-contour.
    cov_test = np.mean(
        [vqr.coverage(Y_test[[i]], X_test[[i]], alpha=alpha) for i in range(N_test)]
    )
    print(f"{cov_test=}")

    # Sample from the fitted conditional distribution, given a specific x.
    Y_sampled = vqr.sample(n=100, x=X_test[0])

    # Calculate coverage on the samples.
    cov_sampled = vqr.coverage(Y_sampled, x=X_test[0], alpha=alpha)
    print(f"{cov_sampled=}")
