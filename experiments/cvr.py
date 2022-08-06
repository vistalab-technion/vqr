from typing import Dict

from numpy import array, argsort
from numpy.linalg import pinv

from vqr import VectorQuantileEstimator
from vqr.api import ScalarQuantileEstimator
from vqr.coverage import measure_width, measure_coverage
from experiments.utils.split import split_train_calib_test
from experiments.datasets.mvn import LinearMVNDataProvider


def experiment(
    X_: array, Y_: array, alpha_low_: float, alpha_high_: float, n_levels: int
) -> Dict[str, float]:
    assert X_.shape[0] == Y_.shape[0]
    assert alpha_low_ < alpha_high_
    assert 0 <= alpha_low_ <= 1
    assert 0 <= alpha_high_ <= 1

    # split train-valid-test
    splits = split_train_calib_test(X_, Y_, split_ratios=(0.15, 0.65))

    # Fit a linear model on train set
    X_train, Y_train = splits["train"]
    A_hat = pinv(X_train.T @ X_train) @ X_train.T @ Y_train

    # Calculate the non-conformity scores on the calibration set
    X_calib, Y_calib = splits["calib"]
    scores_calib = Y_calib - (X_calib @ A_hat)

    # Estimate vector quantiles on the calibration set
    vq = VectorQuantileEstimator(n_levels)
    fitted_vq = vq.fit(scores_calib)
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
    fitted_sq = sq.fit(scores_calib)
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

    # test dataset
    X_test, Y_test = splits["test"]

    vq_coverages_ = []
    sq_coverages_ = []
    for i in range(X_test.shape[0]):
        # prediction set for the data point i
        prediction_set_vq = vq_contour + X_test[i, :] @ A_hat
        prediction_set_sq = sq_contour + X_test[i, :] @ A_hat

        # measure coverage (0/1) for point i
        vq_coverages_.append(measure_coverage(prediction_set_vq, Y_test[i, :][None, :]))
        sq_coverages_.append(measure_coverage(prediction_set_sq, Y_test[i, :][None, :]))

    vq_width_ = measure_width(vq_contour)
    sq_width_ = measure_width(sq_contour)
    vq_coverage_ = sum(vq_coverages_) / len(vq_coverages_)
    sq_coverage_ = sum(sq_coverages_) / len(sq_coverages_)

    results_ = {
        "vq_width": vq_width_,
        "sq_width": sq_width_,
        "vq_coverage": vq_coverage_,
        "sq_coverage": sq_coverage_,
        "realized_alpha_low": T[idx_low],
        "realized_alpha_high": T[idx_high],
        "deflated_alpha_low": T[idx_low],
        "deflated_alpha_high": T[idx_high],
    }

    return results_


if __name__ == "__main__":
    num_trials = 100
    n = 1500
    d = 2
    k = 4
    alpha_low = 0.05
    alpha_high = 0.95
    n_levels = 40
    X, Y = LinearMVNDataProvider(d=d, k=k, seed=42).sample(n=n)

    vq_coverages = []
    vq_widths = []

    sq_coverages = []
    sq_widths = []
    for j in range(num_trials):
        results = experiment(X, Y, alpha_low, alpha_high, n_levels)
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
