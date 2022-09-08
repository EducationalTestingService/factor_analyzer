"""
Factor analysis using MINRES or ML, with optional rotation using Varimax or Promax.

:author: Jeremy Biggs (jeremy.m.biggs@gmail.com)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: Educational Testing Service
:date: 2022-09-05
"""

import warnings

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import minimize
from scipy.stats import chi2, pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_is_fitted

from .rotator import OBLIQUE_ROTATIONS, POSSIBLE_ROTATIONS, Rotator
from .utils import corr, impute_values, partial_correlations, smc

POSSIBLE_SVDS = ["randomized", "lapack"]

POSSIBLE_IMPUTATIONS = ["mean", "median", "drop"]

POSSIBLE_METHODS = ["ml", "mle", "uls", "minres", "principal"]


def calculate_kmo(x):
    """
    Calculate the Kaiser-Meyer-Olkin criterion for items and overall.

    This statistic represents the degree to which each observed variable is
    predicted, without error, by the other variables in the dataset.
    In general, a KMO < 0.6 is considered inadequate.

    Parameters
    ----------
    x : array-like
        The array from which to calculate KMOs.

    Returns
    -------
    kmo_per_variable : :obj:`numpy.ndarray`
        The KMO score per item.
    kmo_total : float
        The overall KMO score.
    """
    # calculate the partial correlations
    partial_corr = partial_correlations(x)

    # calcualte the pair-wise correlations
    x_corr = corr(x)

    # fill matrix diagonals with zeros
    # and square all elements
    np.fill_diagonal(x_corr, 0)
    np.fill_diagonal(partial_corr, 0)

    partial_corr = partial_corr**2
    x_corr = x_corr**2

    # calculate KMO per item
    partial_corr_sum = np.sum(partial_corr, axis=0)
    corr_sum = np.sum(x_corr, axis=0)
    kmo_per_item = corr_sum / (corr_sum + partial_corr_sum)

    # calculate KMO overall
    corr_sum_total = np.sum(x_corr)
    partial_corr_sum_total = np.sum(partial_corr)
    kmo_total = corr_sum_total / (corr_sum_total + partial_corr_sum_total)
    return kmo_per_item, kmo_total


def calculate_bartlett_sphericity(x):
    """
    Compute the Bartlett sphericity test.

    H0: The matrix of population correlations is equal to I.
    H1: The matrix of population correlations is not equal to I.

    The formula for Bartlett's Sphericity test is:

    .. math:: -1 * (n - 1 - ((2p + 5) / 6)) * ln(det(R))

    Where R det(R) is the determinant of the correlation matrix,
    and p is the number of variables.

    Parameters
    ----------
    x : array-like
        The array for which to calculate sphericity.

    Returns
    -------
    statistic : float
        The chi-square value.
    p_value : float
        The associated p-value for the test.
    """
    n, p = x.shape
    x_corr = corr(x)

    corr_det = np.linalg.det(x_corr)
    statistic = -np.log(corr_det) * (n - 1 - (2 * p + 5) / 6)
    degrees_of_freedom = p * (p - 1) / 2
    p_value = chi2.sf(statistic, degrees_of_freedom)
    return statistic, p_value


class FactorAnalyzer(BaseEstimator, TransformerMixin):
    """
    The main exploratory factor analysis class.

    This class:
        (1) Fits a factor analysis model using minres, maximum likelihood,
            or principal factor extraction and returns the loading matrix
        (2) Optionally performs a rotation, with method including:

            (a) varimax (orthogonal rotation)
            (b) promax (oblique rotation)
            (c) oblimin (oblique rotation)
            (d) oblimax (orthogonal rotation)
            (e) quartimin (oblique rotation)
            (f) quartimax (orthogonal rotation)
            (g) equamax (orthogonal rotation)

    Parameters
    ----------
    n_factors : int, optional
        The number of factors to select.
        Defaults to 3.
    rotation : str, optional
        The type of rotation to perform after fitting the factor analysis
        model. If set to ``None``, no rotation will be performed, nor will
        any associated Kaiser normalization.

        Possible values include:

            (a) varimax (orthogonal rotation)
            (b) promax (oblique rotation)
            (c) oblimin (oblique rotation)
            (d) oblimax (orthogonal rotation)
            (e) quartimin (oblique rotation)
            (f) quartimax (orthogonal rotation)
            (g) equamax (orthogonal rotation)

        Defaults to 'promax'.

    method : {'minres', 'ml', 'principal'}, optional
        The fitting method to use, either MINRES or Maximum Likelihood.
        Defaults to 'minres'.
    use_smc : bool, optional
        Whether to use squared multiple correlation as starting guesses for
        factor analysis.
        Defaults to ``True``.
    bounds : tuple, optional
        The lower and upper bounds on the variables for "L-BFGS-B" optimization.
        Defaults to (0.005, 1).
    impute : {'drop', 'mean', 'median'}, optional
        How to handle missing values, if any, in the data: (a) use list-wise
        deletion ('drop'), or (b) impute the column median ('median'), or
        impute the column mean ('mean').
        Defaults to 'median'
    is_corr_matrix : bool, optional
        Set to ``True if the ``data`` is the correlation matrix.
        Defaults to `False`.
    svd_method : {‘lapack’, ‘randomized’}
        The SVD method to use when ``method`` is 'principal'. If 'lapack',
        use standard SVD from ``scipy.linalg``. If 'randomized', use faster
        ``randomized_svd`` function from scikit-learn. The latter should only
        be used if the number of columns is greater than or equal to the
        number of rows in in the dataset.
        Defaults to 'randomized'
    rotation_kwargs, optional
        Dictionary containing keyword arguments for the rotation method.

    Attributes
    ----------
    loadings_ : :obj:`numpy.ndarray`
        The factor loadings matrix.
        ``None``, if ``fit()``` has not been called.
    corr_ : :obj:`numpy.ndarray`
        The original correlation matrix.
        ``None``, if ``fit()`` has not been called.
    rotation_matrix_ : :obj:`numpy.ndarray`
        The rotation matrix, if a rotation has been performed. ``None`` otherwise.
    structure_ : :obj:`numpy.ndarray` or None
        The structure loading matrix. This only exists if ``rotation``
        is 'promax' and is ``None`` otherwise.
    phi_ : :obj:`numpy.ndarray` or None
        The factor correlations matrix. This only exists if ``rotation``
        is 'oblique' and is ``None`` otherwise.

    Notes
    -----
    This code was partly derived from the excellent R package `psych`.

    References
    ----------
    [1] https://github.com/cran/psych/blob/master/R/fa.R

    Examples
    --------
    >>> import pandas as pd
    >>> from factor_analyzer import FactorAnalyzer
    >>> df_features = pd.read_csv('tests/data/test02.csv')
    >>> fa = FactorAnalyzer(rotation=None)
    >>> fa.fit(df_features)
    FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
            method='minres', n_factors=3, rotation=None, rotation_kwargs={},
            use_smc=True)
    >>> fa.loadings_
    array([[-0.12991218,  0.16398154,  0.73823498],
           [ 0.03899558,  0.04658425,  0.01150343],
           [ 0.34874135,  0.61452341, -0.07255667],
           [ 0.45318006,  0.71926681, -0.07546472],
           [ 0.36688794,  0.44377343, -0.01737067],
           [ 0.74141382, -0.15008235,  0.29977512],
           [ 0.741675  , -0.16123009, -0.20744495],
           [ 0.82910167, -0.20519428,  0.04930817],
           [ 0.76041819, -0.23768727, -0.1206858 ],
           [ 0.81533404, -0.12494695,  0.17639683]])
    >>> fa.get_communalities()
    array([0.588758  , 0.00382308, 0.50452402, 0.72841183, 0.33184336,
           0.66208428, 0.61911036, 0.73194557, 0.64929612, 0.71149718])
    """

    def __init__(
        self,
        n_factors=3,
        rotation="promax",
        method="minres",
        use_smc=True,
        is_corr_matrix=False,
        bounds=(0.005, 1),
        impute="median",
        svd_method="randomized",
        rotation_kwargs=None,
    ):
        """Initialize the factor analyzer."""
        self.n_factors = n_factors
        self.rotation = rotation
        self.method = method
        self.use_smc = use_smc
        self.bounds = bounds
        self.impute = impute
        self.is_corr_matrix = is_corr_matrix
        self.svd_method = svd_method
        self.rotation_kwargs = rotation_kwargs

        # default matrices to None
        self.mean_ = None
        self.std_ = None

        self.phi_ = None
        self.structure_ = None

        self.corr_ = None
        self.loadings_ = None
        self.rotation_matrix_ = None
        self.weights_ = None

    def _arg_checker(self):
        """
        Check the input parameters to make sure they're properly formattted.

        We need to do this to ensure that the FactorAnalyzer class can be
        properly cloned when used with grid search CV, for example.
        """
        self.rotation = (
            self.rotation.lower() if isinstance(self.rotation, str) else self.rotation
        )
        if self.rotation not in POSSIBLE_ROTATIONS + [None]:
            raise ValueError(
                f"The rotation must be one of the following: {POSSIBLE_ROTATIONS + [None]}"
            )

        self.method = (
            self.method.lower() if isinstance(self.method, str) else self.method
        )
        if self.method not in POSSIBLE_METHODS:
            raise ValueError(
                f"The method must be one of the following: {POSSIBLE_METHODS}"
            )

        self.impute = (
            self.impute.lower() if isinstance(self.impute, str) else self.impute
        )
        if self.impute not in POSSIBLE_IMPUTATIONS:
            raise ValueError(
                f"The imputation must be one of the following: {POSSIBLE_IMPUTATIONS}"
            )

        self.svd_method = (
            self.svd_method.lower()
            if isinstance(self.svd_method, str)
            else self.svd_method
        )
        if self.svd_method not in POSSIBLE_SVDS:
            raise ValueError(
                f"The SVD method must be one of the following: {POSSIBLE_SVDS}"
            )

        if self.method == "principal" and self.is_corr_matrix:
            raise ValueError(
                "The principal method is only implemented using "
                "the full data set, not the correlation matrix."
            )

        self.rotation_kwargs = (
            {} if self.rotation_kwargs is None else self.rotation_kwargs
        )

    @staticmethod
    def _fit_uls_objective(psi, corr_mtx, n_factors):  # noqa: D401
        """
        The objective function passed for unweighted least-squares (ULS).

        Parameters
        ----------
        psi : array-like
            Value passed to minimize the objective function.
        corr_mtx : array-like
            The correlation matrix.
        n_factors : int
            The number of factors to select.

        Returns
        -------
        error : float
            The scalar error calculated from the residuals of the loading
            matrix.
        """
        np.fill_diagonal(corr_mtx, 1 - psi)

        # get the eigen values and vectors for n_factors
        values, vectors = sp.linalg.eigh(corr_mtx)
        values = values[::-1]

        # this is a bit of a hack, borrowed from R's `fac()` function;
        # if values are smaller than the smallest representable positive
        # number * 100, set them to that number instead.
        values = np.maximum(values, np.finfo(float).eps * 100)

        # sort the values and vectors in ascending order
        values = values[:n_factors]
        vectors = vectors[:, ::-1][:, :n_factors]

        # calculate the loadings
        if n_factors > 1:
            loadings = np.dot(vectors, np.diag(np.sqrt(values)))
        else:
            loadings = vectors * np.sqrt(values[0])

        # calculate the error from the loadings model
        model = np.dot(loadings, loadings.T)

        # note that in a more recent version of the `fa()` source
        # code on GitHub, the minres objective function only sums the
        # lower triangle of the residual matrix; this could be
        # implemented here using `np.tril()` when this change is
        # merged into the stable version of `psych`.
        residual = (corr_mtx - model) ** 2
        error = sp.sum(residual)
        return error

    @staticmethod
    def _normalize_uls(solution, corr_mtx, n_factors):
        """
        Weighted least squares normalization for loadings using MINRES.

        Parameters
        ----------
        solution : array-like
            The solution from the L-BFGS-B optimization.
        corr_mtx : array-like
            The correlation matrix.
        n_factors : int
            The number of factors to select.

        Returns
        -------
        loadings : :obj:`numpy.ndarray`
            The factor loading matrix
        """
        np.fill_diagonal(corr_mtx, 1 - solution)

        # get the eigenvalues and vectors for n_factors
        values, vectors = np.linalg.eigh(corr_mtx)

        # sort the values and vectors in ascending order
        values = values[::-1][:n_factors]
        vectors = vectors[:, ::-1][:, :n_factors]

        # calculate loadings
        # if values are smaller than 0, set them to zero
        loadings = np.dot(vectors, np.diag(np.sqrt(np.maximum(values, 0))))
        return loadings

    @staticmethod
    def _fit_ml_objective(psi, corr_mtx, n_factors):  # noqa: D401
        """
        The objective function for maximum likelihood.

        Parameters
        ----------
        psi : array-like
            Value passed to minimize the objective function.
        corr_mtx : array-like
            The correlation matrix.
        n_factors : int
            The number of factors to select.

        Returns
        -------
        error : float
            The scalar error calculated from the residuals
            of the loading matrix.

        Note
        ----
        The ML objective is based on the `factanal()` function from ``stats``
        package in R. It may generate different results from the ``fa()``
        function in ``psych``.

        References
        ----------
        [1] https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/factanal.R
        """
        sc = np.diag(1 / np.sqrt(psi))
        sstar = np.dot(np.dot(sc, corr_mtx), sc)

        # get the eigenvalues and eigenvectors for n_factors
        values, _ = np.linalg.eigh(sstar)
        values = values[::-1][n_factors:]

        # calculate the error
        error = -(np.sum(np.log(values) - values) - n_factors + corr_mtx.shape[0])
        return error

    @staticmethod
    def _normalize_ml(solution, corr_mtx, n_factors):
        """
        Normalize loadings estimated using maximum likelihood.

        Parameters
        ----------
        solution : array-like
            The solution from the L-BFGS-B optimization.
        corr_mtx : array-like
            The correlation matrix.
        n_factors : int
            The number of factors to select.

        Returns
        -------
        loadings : :obj:`numpy.ndarray`
            The factor loading matrix
        """
        sc = np.diag(1 / np.sqrt(solution))
        sstar = np.dot(np.dot(sc, corr_mtx), sc)

        # get the eigenvalues for n_factors
        values, vectors = np.linalg.eigh(sstar)

        # sort the values and vectors in ascending order
        values = values[::-1][:n_factors]
        vectors = vectors[:, ::-1][:, :n_factors]

        values = np.maximum(values - 1, 0)

        # get the loadings
        loadings = np.dot(vectors, np.diag(np.sqrt(values)))

        return np.dot(np.diag(np.sqrt(solution)), loadings)

    def _fit_principal(self, X):
        """
        Fit factor analysis model using principal factor analysis.

        Parameters
        ----------
        X : array-like
            The full data set.

        Returns
        -------
        loadings : :obj:`numpy.ndarray`
            The factor loadings matrix.
        """
        # standardize the data
        X = X.copy()
        X = (X - X.mean(0)) / X.std(0)

        # if the number of rows is less than the number of columns,
        # warn the user that the number of factors will be constrained
        nrows, ncols = X.shape
        if nrows < ncols and self.n_factors >= nrows:
            warnings.warn(
                "The number of factors will be "
                "constrained to min(n_samples, n_features)"
                "={}.".format(min(nrows, ncols))
            )

        # perform the randomized singular value decomposition
        if self.svd_method == "randomized":
            _, _, V = randomized_svd(X, self.n_factors, random_state=1234567890)
        # otherwise, perform the full SVD
        else:
            _, _, V = np.linalg.svd(X, full_matrices=False)

        corr_mtx = np.dot(X, V.T)
        loadings = np.array([[pearsonr(x, c)[0] for c in corr_mtx.T] for x in X.T])
        return loadings

    def _fit_factor_analysis(self, corr_mtx):
        """
        Fit factor analysis model using either MINRES or maximum likelihood.

        Parameters
        ----------
        corr_mtx : array-like
            The correlation matrix.

        Returns
        -------
        loadings : :obj:`numpy.ndarray`

        Raises
        ------
        ValueError
            If any of the correlations are null, most likely due to having
            zero standard deviation.
        """
        # if `use_smc` is True, get get squared multiple correlations
        # and use these as initial guesses for optimizer
        if self.use_smc:
            smc_mtx = smc(corr_mtx)
            start = (np.diag(corr_mtx) - smc_mtx.T).squeeze()
        # otherwise, just start with a guess of 0.5 for everything
        else:
            start = [0.5 for _ in range(corr_mtx.shape[0])]

        # if `bounds`, set initial boundaries for all variables;
        # this must be a list passed to `minimize()`
        if self.bounds is not None:
            bounds = [self.bounds for _ in range(corr_mtx.shape[0])]
        else:
            bounds = self.bounds

        # minimize the appropriate objective function
        # and the L-BFGS-B algorithm
        if self.method == "ml" or self.method == "mle":
            objective = self._fit_ml_objective
        else:
            objective = self._fit_uls_objective

        # use scipy to perform the actual minimization
        res = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000},
            args=(corr_mtx, self.n_factors),
        )

        if not res.success:
            warnings.warn(f"Failed to converge: {res.message}")

        # transform the final loading matrix (using wls for MINRES,
        # and ml normalization for ML), and convert to DataFrame
        if self.method == "ml" or self.method == "mle":
            loadings = self._normalize_ml(res.x, corr_mtx, self.n_factors)
        else:
            loadings = self._normalize_uls(res.x, corr_mtx, self.n_factors)
        return loadings

    def fit(self, X, y=None):
        """
        Fit factor analysis model using either MINRES, ML, or principal factor analysis.

        By default, use SMC as starting guesses.

        Parameters
        ----------
        X : array-like
            The data to analyze.
        y : ignored

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import FactorAnalyzer
        >>> df_features = pd.read_csv('tests/data/test02.csv')
        >>> fa = FactorAnalyzer(rotation=None)
        >>> fa.fit(df_features)
        FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
                method='minres', n_factors=3, rotation=None, rotation_kwargs={},
                use_smc=True)
        >>> fa.loadings_
        array([[-0.12991218,  0.16398154,  0.73823498],
               [ 0.03899558,  0.04658425,  0.01150343],
               [ 0.34874135,  0.61452341, -0.07255667],
               [ 0.45318006,  0.71926681, -0.07546472],
               [ 0.36688794,  0.44377343, -0.01737067],
               [ 0.74141382, -0.15008235,  0.29977512],
               [ 0.741675  , -0.16123009, -0.20744495],
               [ 0.82910167, -0.20519428,  0.04930817],
               [ 0.76041819, -0.23768727, -0.1206858 ],
               [ 0.81533404, -0.12494695,  0.17639683]])
        """
        # check the input arguments
        self._arg_checker()

        # check if the data is a data frame,
        # so we can convert it to an array
        if isinstance(X, pd.DataFrame):
            X = X.copy().values
        else:
            X = X.copy()

        # now check the array, and make sure it
        # meets all of our expected criteria
        X = check_array(X, force_all_finite="allow-nan", estimator=self, copy=True)

        # check to see if there are any null values, and if
        # so impute using the desired imputation approach
        if np.isnan(X).any() and not self.is_corr_matrix:
            X = impute_values(X, how=self.impute)

        # get the correlation matrix
        if self.is_corr_matrix:
            corr_mtx = X
        else:
            corr_mtx = corr(X)
            self.std_ = np.std(X, axis=0)
            self.mean_ = np.mean(X, axis=0)

        # save the original correlation matrix
        self.corr_ = corr_mtx.copy()

        # fit factor analysis model
        if self.method == "principal":
            loadings = self._fit_principal(X)
        else:
            loadings = self._fit_factor_analysis(corr_mtx)

        # only used if we do an oblique rotations;
        # default rotation matrix to None
        phi = None
        structure = None
        rotation_mtx = None

        # whether to rotate the loadings matrix
        if self.rotation is not None:
            if loadings.shape[1] <= 1:
                warnings.warn(
                    "No rotation will be performed when "
                    "the number of factors equals 1."
                )
            else:
                if "method" in self.rotation_kwargs:
                    warnings.warn(
                        "You cannot pass a rotation method to "
                        "`rotation_kwargs`. This will be ignored."
                    )
                    self.rotation_kwargs.pop("method")
                rotator = Rotator(method=self.rotation, **self.rotation_kwargs)
                loadings = rotator.fit_transform(loadings)
                rotation_mtx = rotator.rotation_
                phi = rotator.phi_
                # update the rotation matrix for everything, except promax
                if self.rotation != "promax":
                    rotation_mtx = np.linalg.inv(rotation_mtx).T

        if self.n_factors > 1:
            # update loading signs to match column sums
            # this is to ensure that signs align with R
            signs = np.sign(loadings.sum(0))
            signs[(signs == 0)] = 1
            loadings = np.dot(loadings, np.diag(signs))

            if phi is not None:
                # update phi, if it exists -- that is, if the rotation is oblique
                # create the structure matrix for any oblique rotation
                phi = np.dot(np.dot(np.diag(signs), phi), np.diag(signs))
                structure = (
                    np.dot(loadings, phi)
                    if self.rotation in OBLIQUE_ROTATIONS
                    else None
                )

        # resort the factors according to their variance,
        # unless the method is principal
        if self.method != "principal":
            variance = self._get_factor_variance(loadings)[0]
            new_order = list(reversed(np.argsort(variance)))
            loadings = loadings[:, new_order].copy()

            # if the structure matrix exists, reorder
            if structure is not None:
                structure = structure[:, new_order].copy()

        self.phi_ = phi
        self.structure_ = structure

        self.loadings_ = loadings
        self.rotation_matrix_ = rotation_mtx
        return self

    def transform(self, X):
        """
        Get factor scores for a new data set.

        Parameters
        ----------
        X : array-like, shape (``n_samples``, ``n_features``)
            The data to score using the fitted factor model.

        Returns
        -------
        X_new : :obj:`numpy.ndarray`, shape (``n_samples``, ``n_components``)
            The latent variables of X.

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import FactorAnalyzer
        >>> df_features = pd.read_csv('tests/data/test02.csv')
        >>> fa = FactorAnalyzer(rotation=None)
        >>> fa.fit(df_features)
        FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
                method='minres', n_factors=3, rotation=None, rotation_kwargs={},
                use_smc=True)
        >>> fa.transform(df_features)
        array([[-1.05141425,  0.57687826,  0.1658788 ],
               [-1.59940101,  0.89632125,  0.03824552],
               [-1.21768164, -1.16319406,  0.57135189],
               ...,
               [ 0.13601554,  0.03601086,  0.28813877],
               [ 1.86904519, -0.3532394 , -0.68170573],
               [ 0.86133386,  0.18280695, -0.79170903]])
        """
        # check if the data is a data frame,
        # so we can convert it to an array
        if isinstance(X, pd.DataFrame):
            X = X.copy().values
        else:
            X = X.copy()

        # now check the array, and make sure it
        # meets all of our expected criteria
        X = check_array(X, force_all_finite=True, estimator=self, copy=True)

        # meets all of our expected criteria
        check_is_fitted(self, "loadings_")

        # see if we saved the original mean and std
        if self.mean_ is None or self.std_ is None:
            warnings.warn(
                "Could not find original mean and standard deviation; using"
                "the mean and standard deviation from the current data set."
            )
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
        else:
            mean = self.mean_
            std = self.std_

        # get the scaled data
        X_scale = (X - mean) / std

        # use the structure matrix, if it exists;
        # otherwise, just use the loadings matrix
        if self.structure_ is not None:
            structure = self.structure_
        else:
            structure = self.loadings_

        try:
            self.weights_ = np.linalg.solve(self.corr_, structure)
        except Exception as error:
            warnings.warn(
                "Unable to calculate the factor score weights; "
                "factor loadings used instead: {}".format(error)
            )
            self.weights_ = self.loadings_

        scores = np.dot(X_scale, self.weights_)
        return scores

    def get_eigenvalues(self):
        """
        Calculate the eigenvalues, given the factor correlation matrix.

        Returns
        -------
        original_eigen_values : :obj:`numpy.ndarray`
            The original eigenvalues
        common_factor_eigen_values : :obj:`numpy.ndarray`
            The common factor eigenvalues

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import FactorAnalyzer
        >>> df_features = pd.read_csv('tests/data/test02.csv')
        >>> fa = FactorAnalyzer(rotation=None)
        >>> fa.fit(df_features)
        FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
                method='minres', n_factors=3, rotation=None, rotation_kwargs={},
                use_smc=True)
        >>> fa.get_eigenvalues()
        (array([ 3.51018854,  1.28371018,  0.73739507,  0.1334704 ,  0.03445558,
                0.0102918 , -0.00740013, -0.03694786, -0.05959139, -0.07428112]),
         array([ 3.51018905,  1.2837105 ,  0.73739508,  0.13347082,  0.03445601,
                0.01029184, -0.0074    , -0.03694834, -0.05959057, -0.07428059]))
        """
        # meets all of our expected criteria
        check_is_fitted(self, ["loadings_", "corr_"])
        corr_mtx = self.corr_.copy()

        e_values, _ = np.linalg.eigh(corr_mtx)
        e_values = e_values[::-1]

        communalities = self.get_communalities()
        communalities = communalities.copy()
        np.fill_diagonal(corr_mtx, communalities)

        values, _ = np.linalg.eigh(corr_mtx)
        values = values[::-1]
        return e_values, values

    def get_communalities(self):
        """
        Calculate the communalities, given the factor loading matrix.

        Returns
        -------
        communalities : :obj:`numpy.ndarray`
            The communalities from the factor loading matrix.

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import FactorAnalyzer
        >>> df_features = pd.read_csv('tests/data/test02.csv')
        >>> fa = FactorAnalyzer(rotation=None)
        >>> fa.fit(df_features)
        FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
                method='minres', n_factors=3, rotation=None, rotation_kwargs={},
                use_smc=True)
        >>> fa.get_communalities()
        array([0.588758  , 0.00382308, 0.50452402, 0.72841183, 0.33184336,
               0.66208428, 0.61911036, 0.73194557, 0.64929612, 0.71149718])
        """
        # meets all of our expected criteria
        check_is_fitted(self, "loadings_")
        loadings = self.loadings_.copy()
        communalities = (loadings**2).sum(axis=1)
        return communalities

    def get_uniquenesses(self):
        """
        Calculate the uniquenesses, given the factor loading matrix.

        Returns
        -------
        uniquenesses : :obj:`numpy.ndarray`
            The uniquenesses from the factor loading matrix.

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import FactorAnalyzer
        >>> df_features = pd.read_csv('tests/data/test02.csv')
        >>> fa = FactorAnalyzer(rotation=None)
        >>> fa.fit(df_features)
        FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
                method='minres', n_factors=3, rotation=None, rotation_kwargs={},
                use_smc=True)
        >>> fa.get_uniquenesses()
        array([0.411242  , 0.99617692, 0.49547598, 0.27158817, 0.66815664,
               0.33791572, 0.38088964, 0.26805443, 0.35070388, 0.28850282])
        """
        # meets all of our expected criteria
        check_is_fitted(self, "loadings_")
        communalities = self.get_communalities()
        communalities = communalities.copy()
        uniqueness = 1 - communalities
        return uniqueness

    @staticmethod
    def _get_factor_variance(loadings):
        """
        Get the factor variances.

        This is a private helper method to get the factor variances,
        because sometimes we need them even before the model is fitted.

        Parameters
        ----------
        loadings : array-like
            The factor loading matrix, in whatever state.

        Returns
        -------
        variance : :obj:`numpy.ndarray`
            The factor variances.
        proportional_variance : :obj:`numpy.ndarray`
            The proportional factor variances.
        cumulative_variances : :obj:`numpy.ndarray`
            The cumulative factor variances.
        """
        n_rows = loadings.shape[0]

        # calculate variance
        loadings = loadings**2
        variance = np.sum(loadings, axis=0)

        # calculate proportional variance
        proportional_variance = variance / n_rows

        # calculate cumulative variance
        cumulative_variance = np.cumsum(proportional_variance, axis=0)

        return (variance, proportional_variance, cumulative_variance)

    def get_factor_variance(self):
        """
        Calculate factor variance information.

        The factor variance information including the variance,
        proportional variance, and cumulative variance for each factor.

        Returns
        -------
        variance : :obj:`numpy.ndarray`
            The factor variances.
        proportional_variance : :obj:`numpy.ndarray`
            The proportional factor variances.
        cumulative_variances : :obj:`numpy.ndarray`
            The cumulative factor variances.

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import FactorAnalyzer
        >>> df_features = pd.read_csv('tests/data/test02.csv')
        >>> fa = FactorAnalyzer(rotation=None)
        >>> fa.fit(df_features)
        FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
                method='minres', n_factors=3, rotation=None, rotation_kwargs={},
                use_smc=True)
        >>> # 1. Sum of squared loadings (variance)
        ... # 2. Proportional variance
        ... # 3. Cumulative variance
        >>> fa.get_factor_variance()
        (array([3.51018854, 1.28371018, 0.73739507]),
         array([0.35101885, 0.12837102, 0.07373951]),
         array([0.35101885, 0.47938987, 0.55312938]))
        """
        # meets all of our expected criteria
        check_is_fitted(self, "loadings_")
        loadings = self.loadings_.copy()
        return self._get_factor_variance(loadings)
