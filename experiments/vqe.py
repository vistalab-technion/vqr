from typing import Dict

from numpy import array, argsort

from vqr.api import ScalarQuantileEstimator, VectorQuantileEstimator
from vqr.coverage import measure_width, measure_coverage
from experiments.utils.split import split_train_calib_test
from experiments.datasets.mvn import IndependentDataProvider
from experiments.datasets.shapes import generate_star, generate_heart


def experiment(
    X_: array, Y_: array, alpha_low_: float, alpha_high_: float, n_levels: int
) -> Dict[str, float]:
    assert X_.shape[0] == Y_.shape[0]
    d = Y_.shape[1]

    # Split train, calibration, and test
    datasets = split_train_calib_test(X_, Y_, split_ratios=(0.0, 0.5))

    # Calibration set
    _, Y_calib = datasets["calib"]

    # Fit vector quantiles
    vq = VectorQuantileEstimator(n_levels)
    fitted_vq = vq.fit(Y_calib)

    # Get the quantiles
    VQ1, VQ2 = fitted_vq.vector_quantiles()
    T = fitted_vq.quantile_levels

    # Construct a quantile contour
    idx_low = argsort(abs(T - alpha_low_))[0]
    idx_high = argsort(abs(T - alpha_high_))[0]

    vq_contour = array(
        [
            [
                *VQ1[idx_low : idx_high + 1, idx_high + 1],
                *VQ1[idx_low, idx_low : idx_high + 1],
                *VQ1[idx_low : idx_high + 1, idx_low],
                *VQ1[idx_high + 1, idx_low : idx_high + 1],
            ],
            [
                *VQ2[idx_low : idx_high + 1, idx_high + 1],
                *VQ2[idx_low, idx_low : idx_high + 1],
                *VQ2[idx_low : idx_high + 1, idx_low],
                *VQ2[idx_high + 1, idx_low : idx_high + 1],
            ],
        ]
    ).T

    # Fit scalar quantiles
    sq = ScalarQuantileEstimator(n_levels)
    fitted_sq = sq.fit(Y_calib)
    SQ = fitted_sq.vector_quantiles()
    SQ1 = SQ[:, 0]
    SQ2 = SQ[:, 1]
    sq_contour = array(
        [
            [
                *SQ1[idx_low : idx_high + 1],
                *SQ1[idx_low : idx_high + 1],
                *[SQ1[idx_low]] * len(SQ2[idx_low : idx_high + 1]),
                *[SQ1[idx_high + 1]] * len(SQ2[idx_low : idx_high + 1]),
            ],
            [
                *[SQ2[idx_high + 1]] * len(SQ1[idx_low : idx_high + 1]),
                *[SQ2[idx_low]] * len(SQ1[idx_low : idx_high + 1]),
                *SQ2[idx_low : idx_high + 1],
                *SQ2[idx_low : idx_high + 1],
            ],
        ]
    ).T

    # Test data
    _, Y_test = datasets["test"]

    # Vector quantile: coverage and width
    vq_coverage_ = measure_coverage(vq_contour, Y_test)
    vq_width_ = measure_width(vq_contour)

    # Scalar quantile: coverage and width
    sq_coverage_ = measure_coverage(sq_contour, Y_test)
    sq_width_ = measure_width(sq_contour)

    results_ = {
        "sq_width": sq_width_,
        "sq_coverage": sq_coverage_,
        "vq_width": vq_width_,
        "vq_coverage": vq_coverage_,
        "realized_alpha_low": T[idx_low],
        "realized_alpha_high": T[idx_high],
        "deflated_alpha_low": T[idx_low],
        "deflated_alpha_high": T[idx_high],
    }

    return results_


if __name__ == "__main__":
    dataset_name = "heart"
    num_trials = 50
    n = 1500
    d_ = 2
    alpha_low = 0.05
    alpha_high = 0.95

    # Generate data
    if dataset_name == "mvn":
        X, Y = IndependentDataProvider(d=d_, k=0).sample(n=n)
    elif dataset_name == "heart":
        X, Y = generate_heart()
    elif dataset_name == "star":
        X, Y = generate_star()
    else:
        X, Y = None, None
        NotImplementedError(f"Unrecognized {dataset_name=}")

    vq_coverages = []
    vq_widths = []

    sq_coverages = []
    sq_widths = []

    for j in range(num_trials):
        results = experiment(X, Y, alpha_low, alpha_high, n_levels=40)
        vq_coverages.append(results["vq_coverage"])
        vq_widths.append(results["vq_width"])

        sq_coverages.append(results["sq_coverage"])
        sq_widths.append(results["sq_width"])

        print(
            f"Trial {j}: \n"
            f"Coverage level: [{round(results['realized_alpha_low'], 3)}, "
            f"{round(results['realized_alpha_high'], 3)}] \n"
            f"Vector quantiles: coverage={round(results['vq_coverage'], 3)}%, "
            f"width={round(results['vq_width'], 3)} \n"
            f"Deflated coverage level: [{round(results['deflated_alpha_low'], 3)}, "
            f"{round(results['deflated_alpha_high'], 3)}] \n"
            f"Scalar quantiles: coverage={round(results['sq_coverage'], 3)}%, "
            f"width={round(results['sq_width'], 3)} \n"
        )

    marginal_vq_coverage = sum(vq_coverages) / len(vq_coverages)
    average_vq_width = sum(vq_widths) / len(vq_widths)

    marginal_sq_coverage = sum(sq_coverages) / len(sq_coverages)
    average_sq_width = sum(sq_widths) / len(sq_widths)

    print(
        f"\t\t     Experiment summary \t\t \n"
        f" \t\t ========================= \t\t \n"
        f"Vector quantiles: \n"
        f"Marginal coverage over {num_trials} trials: "
        f"{marginal_vq_coverage}, average width = {average_vq_width}.\n"
        f"Scalar quantiles: \n"
        f"Marginal coverage over {num_trials} trials: "
        f"{marginal_sq_coverage}, average width = {average_sq_width}."
    )
    print(vq_coverages)
    print(vq_widths)
    print(sq_coverages)
    print(sq_widths)
