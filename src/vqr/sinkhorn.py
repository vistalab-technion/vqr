import numpy as np
from scipy.optimize import broyden1


def sinkhorn_stabilized_vqr(
    a,
    b,
    M,
    reg,
    X=None,
    numItermax=1000,
    tau=1e3,
    stopThr=1e-9,
    warmstart=None,
    verbose=False,
    print_period=20,
    log=False,
    **kwargs,
):
    r"""
    Solve the entropic regularization OT problem with log stabilization

    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)

        s.t. \gamma 1 = a

             \gamma^T 1= b

             \gamma\geq 0
    where :

    - M is the (dim_a, dim_b) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (histograms, both sum to 1)


    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in [2]_ but with the log stabilization
    proposed in [10]_ an defined in [9]_ (Algo 3.1) .


    Parameters
    ----------
    a : ndarray, shape (dim_a,)
        samples weights in the source domain
    b : ndarray, shape (dim_b,)
        samples in the target domain
    M : ndarray, shape (dim_a, dim_b)
        loss matrix
    reg : float
        Regularization term >0
    tau : float
        thershold for max value in u or v for log scaling
    warmstart : tible of vectors
        if given then sarting values for alpha an beta log scalings
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : ndarray, shape (dim_a, dim_b)
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    Examples
    --------

    >>> import ot
    >>> a=[.5,.5]
    >>> b=[.5,.5]
    >>> M=[[0.,1.],[1.,0.]]
    >>> ot.bregman.sinkhorn_stabilized(a, b, M, 1)
    array([[0.36552929, 0.13447071],
           [0.13447071, 0.36552929]])


    References
    ----------

    .. [2] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013

    .. [9] Schmitzer, B. (2016). Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. arXiv preprint arXiv:1610.06519.

    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.


    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    assert warmstart is None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if X is not None:
        X = np.asarray(X, dtype=np.float64)
        X_bar = np.mean(X, axis=0, keepdims=True)
        aX_bar = a[:, None] @ X_bar

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    # test if multiple target
    if len(b.shape) > 1:
        n_hists = b.shape[1]
        a = a[:, np.newaxis]
    else:
        n_hists = 0

    assert n_hists == 0

    # init data
    dim_a = len(a)
    dim_b = len(b)

    cpt = 0
    if log:
        log = {"err": []}

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = np.zeros(dim_a), np.zeros(dim_b)
    else:
        alpha, beta = warmstart

    if n_hists:
        u = np.ones((dim_a, n_hists)) / dim_a
        v = np.ones((dim_b, n_hists)) / dim_b
    else:
        u, v = np.ones(dim_a) / dim_a, np.ones(dim_b) / dim_b
        H = np.ones_like(aX_bar, dtype=np.float64)

    def get_K(alpha, beta):
        """log space computation"""
        return np.exp(-(M - alpha.reshape((dim_a, 1)) - beta.reshape((1, dim_b))) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return np.exp(
            -(M - alpha.reshape((dim_a, 1)) - beta.reshape((1, dim_b))) / reg
            + np.log(u.reshape((dim_a, 1)))
            + np.log(v.reshape((1, dim_b)))
        )

    # print(np.min(K))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1
    while loop:

        uprev = u
        vprev = v
        Hprev = H

        # sinkhorn update
        W = np.exp(H @ X.T / reg)
        KW = K * W
        v = b / (np.dot(KW.T, u) + 1e-16)
        u = a / (np.dot(KW, v) + 1e-16)

        def evaluate_H_constraint(H_):
            return (K * (u[:, None] @ v[None, :]) * np.exp(H_ @ X.T / reg)) @ X - aX_bar

        H = broyden1(evaluate_H_constraint, H)

        # remove numerical problems and store them in K
        if np.abs(u).max() > tau or np.abs(v).max() > tau:
            if n_hists:
                alpha, beta = alpha + reg * np.max(np.log(u), 1), beta + reg * np.max(
                    np.log(v)
                )
            else:
                alpha, beta = alpha + reg * np.log(u), beta + reg * np.log(v)
                if n_hists:
                    u, v = (
                        np.ones((dim_a, n_hists)) / dim_a,
                        np.ones((dim_b, n_hists)) / dim_b,
                    )
                else:
                    u, v = np.ones(dim_a) / dim_a, np.ones(dim_b) / dim_b
            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            if n_hists:
                err_u = abs(u - uprev).max()
                err_u /= max(abs(u).max(), abs(uprev).max(), 1.0)
                err_v = abs(v - vprev).max()
                err_v /= max(abs(v).max(), abs(vprev).max(), 1.0)
                err = 0.5 * (err_u + err_v)
            else:
                transp = get_Gamma(alpha, beta, u, v)
                err = np.linalg.norm((np.sum(transp, axis=0) - b))
            if log:
                log["err"].append(err)

            if verbose:
                if cpt % (print_period * 20) == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(cpt, err))

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        if np.any(np.isnan(u)) or np.any(np.isnan(v)) or np.any(np.isnan(W)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print("Warning: numerical errors at iteration", cpt)
            u = uprev
            v = vprev
            break

        cpt = cpt + 1

    if log:
        if n_hists:
            alpha = alpha[:, None]
            beta = beta[:, None]
        logu = alpha / reg + np.log(u)
        logv = beta / reg + np.log(v)
        log["logu"] = logu
        log["logv"] = logv
        log["alpha"] = alpha + reg * np.log(u)
        log["beta"] = beta + reg * np.log(v)
        log["warmstart"] = (log["alpha"], log["beta"])
        if n_hists:
            res = np.zeros((n_hists))
            for i in range(n_hists):
                res[i] = np.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
            return res, log

        else:
            return get_Gamma(alpha, beta, u, v), log
    else:
        if n_hists:
            res = np.zeros((n_hists))
            for i in range(n_hists):
                res[i] = np.sum(get_Gamma(alpha, beta, u[:, i], v[:, i]) * M)
            return res
        else:
            return get_Gamma(alpha, beta, u, v)
