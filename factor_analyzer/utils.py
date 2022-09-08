"""
Utility functions, used primarily by the confirmatory factor analysis module.

:author: Jeremy Biggs (jeremy.m.biggs@gmail.com)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: Educational Testing Service
:date: 2022-09-05
"""
import warnings

import numpy as np
from scipy.linalg import cholesky


def inv_chol(x, logdet=False):
    """
    Calculate matrix inverse using Cholesky decomposition.

    Optionally, calculate the log determinant of the Cholesky.

    Parameters
    ----------
    x : array-like
        The matrix to invert.
    logdet : bool, optional
        Whether to calculate the log determinant, instead of the inverse.
        Defaults to ``False``.

    Returns
    -------
    chol_inv : array-like
        The inverted matrix.
    chol_logdet : array-like or None
        The log determinant, if ``logdet`` was ``True``, otherwise, ``None``.

    """
    chol = cholesky(x, lower=True)

    chol_inv = np.linalg.inv(chol)
    chol_inv = np.dot(chol_inv.T, chol_inv)
    chol_logdet = None

    if logdet:
        chol_diag = np.diag(chol)
        chol_logdet = np.sum(np.log(chol_diag * chol_diag))

    return chol_inv, chol_logdet


def cov(x, ddof=0):
    """
    Calculate the covariance matrix.

    Parameters
    ----------
    x : array-like
        A 1-D or 2-D array containing multiple variables
        and observations. Each column of x represents a variable,
        and each row a single observation of all those variables.
    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in calculations
        is N - ddof, where N represents the number of elements.
        Defaults to 0.

    Returns
    -------
    r : numpy array
        The covariance matrix of the variables.
    """
    r = np.cov(x, rowvar=False, ddof=ddof)
    return r


def corr(x):
    """
    Calculate the correlation matrix.

    Parameters
    ----------
    x : array-like
        A 1-D or 2-D array containing multiple variables
        and observations. Each column of x represents a variable,
        and each row a single observation of all those variables.

    Returns
    -------
    r : numpy array
        The correlation matrix of the variables.
    """
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0, ddof=0)
    r = cov(x)
    return r


def apply_impute_nan(x, how="mean"):
    """
    Apply a function to impute ``np.nan`` values with the mean or the median.

    Parameters
    ----------
    x : array-like
        The 1-D array to impute.
    how : str, optional
        Whether to impute the 'mean' or 'median'.
        Defaults to 'mean'.

    Returns
    -------
    x : :obj:`numpy.ndarray`
        The array, with the missing values imputed.
    """
    if how == "mean":
        x[np.isnan(x)] = np.nanmean(x)
    elif how == "median":
        x[np.isnan(x)] = np.nanmedian(x)
    return x


def impute_values(x, how="mean"):
    """
    Impute ``np.nan`` values with the mean or median, or drop the containing rows.

    Parameters
    ----------
    x : array-like
        An array to impute.
    how : str, optional
        Whether to impute the 'mean' or 'median'.
        Defaults to 'mean'.

    Returns
    -------
    x : :obj:`numpy.ndarray`
        The array, with the missing values imputed or with rows dropped.
    """
    # impute mean or median, if `how` is set to 'mean' or 'median'
    if how in ["mean", "median"]:
        x = np.apply_along_axis(apply_impute_nan, 0, x, how=how)
    # drop missing if `how` is set to 'drop'
    elif how == "drop":
        x = x[~np.isnan(x).any(1), :].copy()
    return x


def smc(corr_mtx, sort=False):
    """
    Calculate the squared multiple correlations.

    This is equivalent to regressing each variable on all others and
    calculating the r-squared values.

    Parameters
    ----------
    corr_mtx : array-like
        The correlation matrix used to calculate SMC.
    sort : bool, optional
        Whether to sort the values for SMC before returning.
        Defaults to ``False``.

    Returns
    -------
    smc : :obj:`numpy.ndarray`
        The squared multiple correlations matrix.
    """
    corr_inv = np.linalg.inv(corr_mtx)
    smc = 1 - 1 / np.diag(corr_inv)

    if sort:
        smc = np.sort(smc)
    return smc


def covariance_to_correlation(m):
    """
    Compute cross-correlations from the given covariance matrix.

    This is a port of R ``cov2cor()`` function.

    Parameters
    ----------
    m : array-like
        The covariance matrix.

    Returns
    -------
    retval : :obj:`numpy.ndarray`
        The cross-correlation matrix.

    Raises
    ------
    ValueError
        If the input matrix is not square.
    """
    # make sure the matrix is square
    numrows, numcols = m.shape
    if not numrows == numcols:
        raise ValueError("Input matrix must be square")

    Is = np.sqrt(1 / np.diag(m))
    retval = Is * m * np.repeat(Is, numrows).reshape(numrows, numrows)
    np.fill_diagonal(retval, 1.0)
    return retval


def partial_correlations(x):
    """
    Compute partial correlations between variable pairs.

    This is a python port of the ``pcor()`` function implemented in
    the ``ppcor`` R package, which computes partial correlations
    for each pair of variables in the given array, excluding all
    other variables.

    Parameters
    ----------
    x : array-like
        An array containing the feature values.

    Returns
    -------
    pcor : :obj:`numpy.ndarray`
        An array containing the partial correlations of of each
        pair of variables in the given array, excluding all other
        variables.
    """
    numrows, numcols = x.shape
    x_cov = cov(x, ddof=1)
    # create empty array for when we cannot compute the
    # matrix inversion
    empty_array = np.empty((numcols, numcols))
    empty_array[:] = np.nan
    if numcols > numrows:
        icvx = empty_array
    else:
        # if the determinant is less than the lowest representable
        # 32 bit integer, then we use the pseudo-inverse;
        # otherwise, use the inverse; if a linear algebra error
        # occurs, then we just set the matrix to empty
        try:
            assert np.linalg.det(x_cov) > np.finfo(np.float32).eps
            icvx = np.linalg.inv(x_cov)
        except AssertionError:
            icvx = np.linalg.pinv(x_cov)
            warnings.warn(
                "The inverse of the variance-covariance matrix "
                "was calculated using the Moore-Penrose generalized "
                "matrix inversion, due to its determinant being at "
                "or very close to zero."
            )
        except np.linalg.LinAlgError:
            icvx = empty_array

    pcor = -1 * covariance_to_correlation(icvx)
    np.fill_diagonal(pcor, 1.0)
    return pcor


def unique_elements(seq):
    """
    Get first unique instance of every list element, while maintaining order.

    Parameters
    ----------
    seq : list-like
        The list of elements.

    Returns
    -------
    seq : list
        The updated list of elements.
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def fill_lower_diag(x):
    """
    Fill the lower diagonal of a square matrix, given a 1-D input array.

    Parameters
    ----------
    x : array-like
        The flattened input matrix that will be used to fill the lower
        diagonal of the square matrix.

    Returns
    -------
    out : :obj:`numpy.ndarray`
        The output square matrix, with the lower diagonal filled by x.

    References
    ----------
    [1] https://stackoverflow.com/questions/51439271/
        convert-1d-array-to-lower-triangular-matrix
    """
    x = np.array(x)
    x = x if len(x.shape) == 1 else np.squeeze(x, axis=1)
    n = int(np.sqrt(len(x) * 2)) + 1
    out = np.zeros((n, n), dtype=float)
    out[np.tri(n, dtype=bool, k=-1)] = x
    return out


def merge_variance_covariance(variances, covariances=None):
    """
    Merge variances and covariances into a single variance-covariance matrix.

    Parameters
    ----------
    variances : array-like
        The variances that will be used to fill the diagonal of the
        square matrix.
    covariances : array-like or None, optional
        The flattened input matrix that will be used to fill the lower and
        upper diagonal of the square matrix. If None, then only the variances
        will be used.
        Defaults to ``None``.

    Returns
    -------
    variance_covariance : :obj:`numpy.ndarray`
        The variance-covariance matrix.
    """
    variances = (
        variances if len(variances.shape) == 1 else np.squeeze(variances, axis=1)
    )
    if covariances is None:
        variance_covariance = np.zeros((variances.shape[0], variances.shape[0]))
    else:
        variance_covariance = fill_lower_diag(covariances)
        variance_covariance += variance_covariance.T
    np.fill_diagonal(variance_covariance, variances)
    return variance_covariance


def get_first_idxs_from_values(x, eq=1, use_columns=True):
    """
    Get the indexes  for a given value.

    Parameters
    ----------
    x : array-like
        The input matrix.
    eq : str or int, optional
        The given value to find.
        Defaults to 1.
    use_columns : bool, optional
        Whether to get the first indexes using the columns.
        If ``False``, then use the rows instead.
        Defaults to ``True``.

    Returns
    -------
    row_idx : list
        A list of row indexes.
    col_idx : list
        A list of column indexes.
    """
    x = np.array(x)
    if use_columns:
        n = x.shape[1]
        row_idx = [np.where(x[:, i] == eq)[0][0] for i in range(n)]
        col_idx = list(range(n))
    else:
        n = x.shape[0]
        col_idx = [np.where(x[i, :] == eq)[0][0] for i in range(n)]
        row_idx = list(range(n))
    return row_idx, col_idx


def get_free_parameter_idxs(x, eq=1):
    """
    Get the free parameter indices from the flattened matrix.

    Parameters
    ----------
    x : array-like
        The input matrix.
    eq : str or int, optional
        The value that free parameters should be equal to. ``np.nan`` fields
        will be populated with this value.
        Defaults to 1.

    Returns
    -------
    idx : :obj:`numpy.ndarray`
        The free parameter indexes.
    """
    x[np.isnan(x)] = eq
    x = x.flatten(order="F")
    return np.where(x == eq)[0]


def duplication_matrix(n=1):
    """
    Calculate the duplication matrix.

    A function to create the duplication matrix (Dn), which is
    the unique n2 × n(n+1)/2 matrix which, for any n × n symmetric
    matrix A, transforms vech(A) into vec(A), as in Dn vech(A) = vec(A).

    Parameters
    ----------
    n : int, optional
        The dimension of the n x n symmetric matrix.
        Defaults to 1.

    Returns
    -------
    duplication_matrix : :obj:`numpy.ndarray`
        The duplication matrix.

    Raises`
    ------
    ValueError
        If ``n`` is not a positive integer greater than 1.

    References
    ----------
    https://en.wikipedia.org/wiki/Duplication_and_elimination_matrices
    """
    if n < 1:
        raise ValueError(
            "The argument `n` must be a " "positive integer greater than 1."
        )

    dup = np.zeros((int(n * n), int(n * (n + 1) / 2)))
    count = 0
    for j in range(n):
        dup[j * n + j, count + j] = 1
        if j < n - 1:
            for i in range(j + 1, n):
                dup[j * n + i, count + i] = 1
                dup[i * n + j, count + i] = 1
        count += n - j - 1
    return dup


def duplication_matrix_pre_post(x):
    """
    Transform given input symmetric matrix using pre-post duplication.

    Parameters
    ----------
    x : array-like
        The input matrix.

    Returns
    -------
    out : :obj:`numpy.ndarray`
        The transformed matrix.

    Raises
    ------
    AssertionError
        If ``x`` is not symmetric.
    """
    assert x.shape[0] == x.shape[1]

    n2 = x.shape[1]
    n = int(np.sqrt(n2))

    idx1 = get_symmetric_lower_idxs(n)
    idx2 = get_symmetric_upper_idxs(n)

    out = x[idx1, :] + x[idx2, :]
    u = np.where([i in idx2 for i in idx1])[0]
    out[u, :] = out[u, :] / 2.0
    out = out[:, idx1] + out[:, idx2]
    out[:, u] = out[:, u] / 2.0
    return out


def commutation_matrix(p, q):
    """
    Calculate the commutation matrix.

    This matrix transforms the vectorized form of the matrix into the
    vectorized form of its transpose.

    Parameters
    ----------
    p : int
        The number of rows.
    q : int
        The number of columns.

    Returns
    -------
    commutation_matrix : :obj:`numpy.ndarray`
        The commutation matrix

    References
    ----------
    https://en.wikipedia.org/wiki/Commutation_matrix
    """
    identity = np.eye(p * q)
    indices = np.arange(p * q).reshape((p, q), order="F")
    return identity.take(indices.ravel(), axis=0)


def get_symmetric_lower_idxs(n=1, diag=True):
    """
    Get the indices for the lower triangle of a symmetric matrix.

    Parameters
    ----------
    n : int, optional
        The dimension of the n x n symmetric matrix.
        Defaults to 1.
    diag : bool, optional
        Whether to include the diagonal.

    Returns
    -------
    indices : :obj:`numpy.ndarray`
        The indices for the lower triangle.
    """
    rows = np.repeat(np.arange(n), n).reshape(n, n)
    cols = rows.T
    if diag:
        return np.where((rows >= cols).T.flatten())[0]
    return np.where((cols > rows).T.flatten())[0]


def get_symmetric_upper_idxs(n=1, diag=True):
    """
    Get the indices for the upper triangle of a symmetric matrix.

    Parameters
    ----------
    n : int, optional
        The dimension of the n x n symmetric matrix.
        Defaults to 1.
    diag : bool, optional
        Whether to include the diagonal.

    Returns
    -------
    indices : :obj:`numpy.ndarray`
        The indices for the upper triangle.
    """
    rows = np.repeat(np.arange(n), n).reshape(n, n)
    cols = rows.T
    temp = np.arange(n * n).reshape(n, n)
    if diag:
        return temp.T[(rows >= cols).T]
    return temp.T[(cols > rows).T]
