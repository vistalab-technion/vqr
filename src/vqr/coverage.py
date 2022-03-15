import numpy as np
from numpy.typing import ArrayLike as Array
from scipy.spatial import ConvexHull


def measure_coverage(quantile_contour: Array, data: Array) -> float:
    """
    Measures the proportion of given points that lie in side a given surface.
    The surface is approximated by its convex hull, all points are checked
    if they lie within the surface or not.

    :param quantile_contour: points representing a quantile contour, of shape (M, d)
    :param data: the points that need to be checked if they are inliers, of shape (N, d)
    :return: Proportion of points that lie within the given contour.
    """

    if np.ndim(quantile_contour) != 2 or np.ndim(data) != 2:
        raise ValueError("Both input arrays must be 2d")

    N, d = data.shape

    if d == 1:
        coverage_ = (data >= quantile_contour[0]) & (data <= quantile_contour[-1])
    else:
        cvx_hull = ConvexHull(quantile_contour)
        coverage_ = np.array([point_in_hull(d, cvx_hull) for d in data])  # type: ignore

    return sum(coverage_).item() / len(coverage_)


def measure_width(quantile_contour: Array) -> float:
    """
    Measures the volume of the convex hull of the given array of points.
    :param quantile_contour: Points constituting the boundary of the contour over
    which the convex hull needs to be constructed, of shape (M, d).
    :return: The area (in 2D) / volume (in 3D) of the convex hull of the given points.
    """
    if np.ndim(quantile_contour) != 2:
        raise ValueError("Input must be 2d")

    _, d = quantile_contour.shape

    if d == 1:
        return quantile_contour[-1] - quantile_contour[0]

    cvx_hull = ConvexHull(quantile_contour)
    volume: float = cvx_hull.volume  # type: ignore
    return volume


def point_in_hull(point: Array, hull: ConvexHull, tolerance=1e-12) -> bool:
    """
    Returns whether a point is contained inside a convex hull.
    :param point: A point in n-d.
    :param hull: A ConvexHull object.
    :param tolerance: Tolerance for comparison.
    :return: True if the point is contained.
    """
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations  # type:ignore
    )
