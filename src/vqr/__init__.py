# isort: skip_file
from vqr.cvqf import QuantileFunction
from vqr.solvers import VQRSolver, VQRSolution
from vqr.api import VectorQuantileEstimator, VectorQuantileRegressor
from pkg_resources import DistributionNotFound, get_distribution

try:
    dist_name = "vqr"
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound
