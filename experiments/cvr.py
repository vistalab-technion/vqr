from typing import Dict

from numpy import array
from numpy.linalg import pinv

from vqr import VectorQuantileEstimator
from vqr.api import ScalarQuantileEstimator
from vqr.data import split_train_calib_test, generate_linear_x_y_mvn_data
from vqr.coverage import measure_width, measure_coverage


def experiment(X_: array, Y_: array, n_levels: int) -> Dict[str, float]:
    assert X.shape[0] == Y.shape[0]
    d = Y.shape[1]

    # split train-valid-test
    splits = split_train_calib_test(X_, Y_, split_ratios=(0.35, 0.65))

    # Fit a linear model on train set
    X_train, Y_train = splits["train"]
    A_hat = pinv(X_train.T @ X_train) @ X_train.T @ Y_train

    # Calculate the non-conformity scores on the calibration set
    X_calib, Y_calib = splits["calib"]
    scores_calib = Y_calib - (X_calib @ A_hat)

    # Estimate vector quantiles on the calibration set
    vq = VectorQuantileEstimator(n_levels)
    fitted_vq = vq.fit(scores_calib)
    VQ1, VQ2 = fitted_vq.quantile_values
    VQ1 = VQ1.reshape([n_levels, n_levels])
    VQ2 = VQ2.reshape([n_levels, n_levels])

    # Construct a quantile vq_surface
    # When t = 40, 1:-2 is equivalent to asking 0.05 - 0.95.
    # Thus we expect 90% coverage
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
    fitted_sq = sq.fit(scores_calib)
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

    # test dataset
    X_test, Y_test = splits["test"]

    vq_coverages_ = []
    sq_coverages_ = []
    for i in range(X_test.shape[0]):
        # prediction set for the data point i
        prediction_set_vq = vq_surface + X_test[i, :] @ A_hat
        prediction_set_sq = sq_surface + X_test[i, :] @ A_hat

        # measure coverage (0/1) for point i
        vq_coverages_.append(measure_coverage(prediction_set_vq, Y_test[i, :][None, :]))
        sq_coverages_.append(measure_coverage(prediction_set_sq, Y_test[i, :][None, :]))

    vq_width_ = measure_width(vq_surface)
    sq_width_ = measure_width(sq_surface)
    vq_coverage_ = sum(vq_coverages_) / len(vq_coverages_)
    sq_coverage_ = sum(sq_coverages_) / len(sq_coverages_)

    results_ = {
        "vq_width": vq_width_,
        "sq_width": sq_width_,
        "vq_coverage": vq_coverage_,
        "sq_coverage": sq_coverage_,
    }

    return results_


if __name__ == "__main__":
    num_trials = 10
    n = 2000
    d = 2
    k = 5
    X, Y = generate_linear_x_y_mvn_data(n=n, d=d, k=k)

    vq_coverages = []
    vq_widths = []

    sq_coverages = []
    sq_widths = []
    for j in range(num_trials):
        results = experiment(X, Y, n_levels=40)
        vq_coverages.append(results["vq_coverage"])
        vq_widths.append(results["vq_width"])

        sq_coverages.append(results["sq_coverage"])
        sq_widths.append(results["sq_width"])

        print(
            f"Trial {j}: \n"
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
        f"\t\t     Experiment summary \t\t \n"
        f" \t\t ========================= \t\t \n"
        f"Vector quantiles: \n"
        f"Marginal coverage over {num_trials} trials: "
        f"{marginal_vq_coverage}, average width = {average_vq_width}.\n"
        f"Scalar quantiles: \n"
        f"Marginal coverage over {num_trials} trials: "
        f"{marginal_sq_coverage}, average width = {average_sq_width}."
    )
