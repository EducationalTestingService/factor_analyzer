"""
Utility functions, used primarily by
the confirmatory factor analysis module.

:author: Jeremy Biggs (jbiggs@ets.org)
:date: 2/05/2019
:organization: ETS
"""
import numpy as np
import pandas as pd


def unique_elements(seq):
    """
    Get the first unique instance of every
    element of the list, while maintaining
    the original order of those instances.

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
    Fill the lower diagonal of a square matrix,
    given a 1d input array.

    Parameters
    ----------
    x : np.array
        The flattened input matrix that will be used to fill
        the lower diagonal of the square matrix.

    Returns
    -------
    out : np.array
        The output square matrix, with the lower
        diagonal filled by x.

    Reference
    ---------
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
    Merge the variance and covariances into a single
    variance-covariance matrix.

    Parameters
    ----------
    variances : np.array
        The variances that will be used to fill the diagonal
        of the square matrix.
    covariances : np.array or None, optional
        The flattened input matrix that will be used to fill
        the lower and upper diagonal of the square matrix. If
        None, then only the variances will be used.
        Defaults to None.

    Returns
    -------
    variance_covariance : np.array
        The variance-covariance matrix.
    """
    variances = np.array(variances)
    variances = (variances if len(variances.shape) == 1
                 else np.squeeze(variances, axis=1))
    if covariances is None:
        variance_covariance = np.zeros((variances.shape[0],
                                        variances.shape[0]))
    else:
        covariances = np.array(covariances)
        variance_covariance = fill_lower_diag(covariances)
        variance_covariance += variance_covariance.T
    np.fill_diagonal(variance_covariance, variances)
    return variance_covariance


def get_first_idxs_from_values(x, eq=1, use_columns=True):
    """
    Get the fixed index

    Parameters
    ----------
    x : np.array
        The input matrix.
    eq : str or int, optional
        The given value to find.
        Defaults to 1.
    use_columns : bool, optional
        Whether to get the first indexes using
        The columns. If False, then use the rows
        instead.
        Defaults to True

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


def get_free_parameter_idxs(x, eq='X'):
    """
    Get the free parameter indexes from
    the flattened matrix.

    Parameters
    ----------
    x : np.array
        The input matrix.
    eq : str or int, optional
        The value that free parameters
        should be equal to. NaN fields
        will be populated with this value.
        Defaults to 'X'
    """
    x = pd.DataFrame(x).fillna(eq)
    x = x.values.flatten(order='F')
    return np.where(x == eq)[0]


def duplication_matrix(n=1):
    """
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
    duplication_matrix : np.array
        The duplication matrix.

    Raises`
    ------
    ValueError
        If `n` is not a positive integer greater than 1.

    References
    ----------
    https://en.wikipedia.org/wiki/Duplication_and_elimination_matrices
    """
    if n < 1:
        raise ValueError('The argument `n` must be a '
                         'positive integer greater than 1.')

    dup = np.zeros((int(n * n), int(n * (n + 1) / 2)))
    count = 0
    for j in range(n):
        dup[j * n + j, count + j] = 1
        if (j < n - 1):
            for i in range(j + 1, n):
                dup[j * n + i, count + i] = 1
                dup[i * n + j, count + i] = 1
        count += n - j - 1
    return dup


def duplication_matrix_pre_post(x):
    """
    Given an input symmetric matrix,
    transform perform the pre-post duplication.

    Parameters
    ----------
    x : np.array
        The input matrix

    Returns
    -------
    out : pd.DataFrame
        The transformed matrix.

    Raises
    ------
    AssertionError
        If `x` is not symmetric.
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
    return pd.DataFrame(out)


def commutation_matrix(p, q):
    """
    Calculate the commutation matrix, which transforms
    the vectorized form of the matrix into the vectorized
    form of its transpose.

    Parameters
    ----------
    p : int
        The number of rows.
    q : int
        The number of columns.

    Returns
    -------
    commutation_matrix : np.array
        The commutation matrix

    References
    ----------
    https://en.wikipedia.org/wiki/Commutation_matrix
    """
    identity = np.eye(p * q)
    indices = np.arange(p * q).reshape((p, q), order='F')
    return identity.take(indices.ravel(), axis=0)


def get_symmetric_lower_idxs(n=1, diag=True):
    """
    Get the indexes for the lower triangle of
    a symmetric matrix.

    Parameters
    ----------
    n : int, optional
        The dimension of the n x n symmetric matrix.
        Defaults to 1.
    diag : bool, optional
        Whether to include the diagonal.

    Returns
    -------
    indexes : np.array
        The indexes for the lower triangle.
    """
    rows = np.repeat(np.arange(n), n).reshape(n, n)
    cols = rows.T
    if diag:
        return np.where((rows >= cols).T.flatten())[0]
    return np.where((cols > rows).T.flatten())[0]


def get_symmetric_upper_idxs(n=1, diag=True):
    """
    Get the indexes for the upper triangle of
    a symmetric matrix.

    Parameters
    ----------
    n : int, optional
        The dimension of the n x n symmetric matrix.
        Defaults to 1.
    diag : bool, optional
        Whether to include the diagonal.

    Returns
    -------
    indexes : np.array
        The indexes for the upper triangle.
    """

    rows = np.repeat(np.arange(n), n).reshape(n, n)
    cols = rows.T
    temp = np.arange(n * n).reshape(n, n)
    if diag:
        return temp.T[(rows >= cols).T]
    return temp.T[(cols > rows).T]
