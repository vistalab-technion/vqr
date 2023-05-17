from vqr.api import VectorQuantileRegressor
from experiments.data.mvn import LinearMVNDataProvider
from vqr.solvers.dual.regularized_lse import RegularizedDualVQRSolver

n = 10000
k = 50
d = 2
data_provider = LinearMVNDataProvider(d=d, k=k, seed=42)
X, Y = data_provider.sample(n=n)
epsilon = 1e-3
num_epochs = 20000
T = 32

solver = RegularizedDualVQRSolver(
    verbose=True,
    num_epochs=num_epochs,
    epsilon=epsilon,
    lr=2.9,
    gpu=True,
    full_precision=False,
    lr_factor=0.9,
    lr_patience=500,
    lr_threshold=0.5 * 0.01,
)

vqr = VectorQuantileRegressor(n_levels=T, solver=solver)
vqr.fit(X, Y)
