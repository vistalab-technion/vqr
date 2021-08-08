from typing import Dict

from numpy import array

from vqr.api import ScalarQuantileEstimator, VectorQuantileEstimator
from vqr.data import generate_mvn_data, split_train_calib_test
from vqr.coverage import measure_width, measure_coverage


def experiment(X_: array, Y_: array, n_levels: int) -> Dict[str, float]:
    assert X.shape[0] == Y.shape[0]
    d = Y.shape[1]

    # Split train, calibration, and test
    datasets = split_train_calib_test(X_, Y_, split_ratios=(0.0, 0.5))

    # Calibration set
    _, Y_calib = datasets["calib"]

    # Fit vector quantiles
    vq = VectorQuantileEstimator(n_levels)
    fitted_vq = vq.fit(Y_calib)

    # Get the quantiles
    VQ1, VQ2 = fitted_vq.quantile_values

    # Construct a quantile surface
    vq_surface = (
        array(
            [
                [*VQ1[1:-2, -2], *VQ1[1, 1:-2], *VQ1[1:-2, 1], *VQ1[-2, 1:-2]],
                [*VQ2[1:-2, -2], *VQ2[1, 1:-2], *VQ2[1:-2, 1], *VQ2[-2, 1:-2]],
            ]
        ).T
        * (n_levels ** d)
    )

    # Fit scalar quantiles
    sq = ScalarQuantileEstimator(n_levels)
    fitted_sq = sq.fit(Y_calib)
    SQ = fitted_sq.quantile_values
    SQ1 = SQ[:, 0]
    SQ2 = SQ[:, 1]
    sq_surface = array(
        [
            [
                *SQ1[1:-1],
                *SQ1[1:-1],
                *[SQ1[1]] * len(SQ2[1:-1]),
                *[SQ1[-1]] * len(SQ2[1:-1]),
            ],
            [
                *[SQ2[-1]] * len(SQ1[1:-1]),
                *[SQ2[1]] * len(SQ1[1:-1]),
                *SQ2[1:-1],
                *SQ2[1:-1],
            ],
        ]
    ).T

    # Test data
    _, Y_test = datasets["test"]

    # Vector quantile: coverage and width
    vq_coverage_ = measure_coverage(vq_surface, Y_test)
    vq_width_ = measure_width(vq_surface)

    # Scalar quantile: coverage and width
    sq_coverage_ = measure_coverage(sq_surface, Y_test)
    sq_width_ = measure_width(sq_surface)

    results_ = {
        "sq_width": sq_width_,
        "sq_coverage": sq_coverage_,
        "vq_width": vq_width_,
        "vq_coverage": vq_coverage_,
    }

    return results_


if __name__ == "__main__":
    num_trials = 10
    d_ = 2
    n = 2000
    # Generate data
    X, Y = generate_mvn_data(n=n, d=d_, k=0)

    vq_coverages = []
    vq_widths = []

    sq_coverages = []
    sq_widths = []

    for i in range(num_trials):
        results = experiment(X, Y, n_levels=40)
        vq_coverages.append(results["vq_coverage"])
        vq_widths.append(results["vq_width"])

        sq_coverages.append(results["sq_coverage"])
        sq_widths.append(results["sq_width"])

        print(
            f"Trial {i}: \n"
            f"Vector quantiles: coverage={results['vq_coverage']}%, "
            f"width={results['vq_width']} \n"
            f"Scalar quantiles: coverage={results['sq_coverage']}%, "
            f"width={results['sq_width']} \n"
        )

    marginal_vq_coverage = sum(vq_coverages) / len(vq_coverages)
    average_vq_width = sum(vq_widths) / len(vq_widths)

    marginal_sq_coverage = sum(sq_coverages) / len(sq_coverages)
    average_sq_width = sum(sq_widths) / len(sq_widths)

    print(
        f"\t\t Experiment summary \t\t \n"
        f" \t\t ======================== \t \t"
        f"Vector quantiles: \n"
        f"Marginal coverage over {num_trials} trials: "
        f"{marginal_vq_coverage}, average width = {average_vq_width}.\n"
        f"Scalar quantiles: \n"
        f"Marginal coverage over {num_trials} trials: "
        f"{marginal_sq_coverage}, average width = {average_sq_width}."
    )
