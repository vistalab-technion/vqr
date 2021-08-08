from numpy import dot, array
from scipy.spatial import ConvexHull


def measure_coverage(quantile_surface: array, data: array) -> float:
    """
    Measures the % of given points that lie in side a given surface.
    The surface is approximated by its convex hull, all points are checked
    if they lie within the surface or not.

    :param quantile_surface: points representing the boundary of the surface
    :param data: the points that need to be checked if they are in-liers.
    :return: % of points that lie within the given surface.
    """

    def point_in_hull(point, hull, tolerance=1e-12):
        return all((dot(eq[:-1], point) + eq[-1] <= tolerance) for eq in hull.equations)

    cvx_hull = ConvexHull(quantile_surface)
    coverage_ = [point_in_hull(d, cvx_hull) for d in data]  # type: ignore
    return (sum(coverage_) / len(coverage_)) * 100


def measure_width(quantile_surface: array) -> float:
    """
    Measures the volume of the convex hull of the given array of points.
    :param quantile_surface: Points constituting the boundary of the surface over
    which the convex hull needs to be constructed.
    :return: The area (in 2D) / volume (in 3D) of the convex hull of the given points.
    """
    cvx_hull = ConvexHull(quantile_surface)
    volume: float = cvx_hull.volume  # type: ignore
    return volume
