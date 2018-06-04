"""
Factor analysis using MINRES or ML,
with optional rotation using Varimax or Promax.

:author: Jeremy Biggs (jbiggs@ets.org)

:date: 10/25/2017
:organization: ETS
"""

import logging
import warnings

import numpy as np
import scipy as sp
import pandas as pd

from scipy.stats import chi2
from scipy.optimize import minimize

from factor_analyzer.rotator import Rotator
from factor_analyzer.rotator import POSSIBLE_ROTATIONS, OBLIQUE_ROTATIONS


def covariance_to_correlation(m):
    """
    This is a port of the R `cov2cor` function.

    Parameters
    ----------
    m : numpy array
        The covariance matrix.

    Returns
    -------
    retval : numpy array
        The cross-correlation matrix.

    Raises
    ------
    ValueError
        If the input matrix is not square.
    """

    # make sure the matrix is square
    numrows, numcols = m.shape
    if not numrows == numcols:
        raise ValueError('Input matrix must be square')

    Is = np.sqrt(1 / np.diag(m))
    retval = Is * m * np.repeat(Is, numrows).reshape(numrows, numrows)
    np.fill_diagonal(retval, 1.0)
    return retval


def partial_correlations(data):
    """
    This is a python port of the `pcor` function implemented in
    the `ppcor` R package, which computes partial correlations
    of each pair of variables in the given data frame `data`,
    excluding all other variables.

    Parameters
    ----------
    data : pd.DataFrame
        Data frame containing the feature values.

    Returns
    -------
    df_pcor : pd.DataFrame
        Data frame containing the partial correlations of of each
        pair of variables in the given data frame `df`,
        excluding all other variables.
    """
    numrows, numcols = data.shape
    df_cov = data.cov()
    columns = df_cov.columns

    # return a matrix of nans if the number of columns is
    # greater than the number of rows. When the ncol == nrows
    # we get the degenerate matrix with 1 only. It is not meaningful
    # to compute partial correlations when ncol > nrows.

    # create empty array for when we cannot compute the
    # matrix inversion
    empty_array = np.empty((len(columns), len(columns)))
    empty_array[:] = np.nan
    if numcols > numrows:
        icvx = empty_array
    else:
        # we also return nans if there is singularity in the data
        # (e.g. all human scores are the same)
        try:
            icvx = np.linalg.inv(df_cov)
        except np.linalg.LinAlgError:
            icvx = empty_array
    pcor = -1 * covariance_to_correlation(icvx)
    np.fill_diagonal(pcor, 1.0)
    df_pcor = pd.DataFrame(pcor, columns=columns, index=columns)
    return df_pcor


def calculate_kmo(data):
    """
    Calculate the Kaiser-Meyer-Olkin criterion
    for items and overall. This statistic represents
    the degree to which each observed variable is
    predicted, without error, by the other variables
    in the dataset. In general, a KMO < 0.6 is considered
    inadequate.

    Parameters
    ----------
    data : pd.DataFrame
        The data frame from which to calculate KMOs.

    Returns
    -------
    kmo_per_variable : pd.DataFrame
        The KMO score per item.
    kmo_total : float
        The KMO score overall.
    """

    # calculate the partial correlations
    partial_corr = partial_correlations(data)
    partial_corr = partial_corr.values

    # calcualte the pair-wise correlations
    corr = data.corr()
    corr = corr.values

    # fill matrix diagonals with zeros
    # and square all elements
    np.fill_diagonal(corr, 0)
    np.fill_diagonal(partial_corr, 0)

    partial_corr = partial_corr**2
    corr = corr**2

    # calculate KMO per item
    partial_corr_sum = partial_corr.sum(0)
    corr_sum = corr.sum(0)
    kmo_per_item = corr_sum / (corr_sum + partial_corr_sum)
    kmo_per_item = pd.DataFrame(kmo_per_item,
                                index=data.columns,
                                columns=['KMO'])

    # calculate KMO overall
    corr_sum_total = corr.sum()
    partial_corr_sum_total = partial_corr.sum()
    kmo_total = corr_sum_total / (corr_sum_total + partial_corr_sum_total)
    return kmo_per_item, kmo_total


def calculate_bartlett_sphericity(data):
    """
    Test the hypothesis that the correlation matrix
    is equal to the identity matrix.identity

    H0: The matrix of population correlations is equal to I.
    H1: The matrix of population correlations is not equal to I.

    The formula for Bartlett's Sphericity test is:

    .. math:: -1 * (n - 1 - ((2p + 5) / 6)) * ln(det(R))

    Where R det(R) is the determinant of the correlation matrix,
    and p is the number of variables.

    Parameters
    ----------
    data : pd.DataFrame
        The data frame from which to calculate sphericity.

    Returns
    -------
    statistic : float
        The chi-square value.
    p_value : float
        The associated p-value for the test.
    """
    n, p = data.shape

    corr = data.corr()

    corr_det = np.linalg.det(corr)
    statistic = -np.log(corr_det) * (n - 1 - (2 * p + 5) / 6)
    degrees_of_freedom = p * (p - 1) / 2
    p_value = chi2.pdf(statistic, degrees_of_freedom)
    return statistic, p_value


class FactorAnalyzer:
    """
    A FactorAnalyzer class, which -
        (1) Fits a factor analysis model using minres or maximum likelihood,
            and returns the loading matrix
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
    log_warnings : bool
        Whether to log warnings, such as failure to
        converge.
        Defaults to False.

    Attributes
    ----------
    loadings : pd.DataFrame
        The factor loadings matrix.
        Default to None, if `analyze()` has not
        been called.
    corr : pd.DataFrame
        The original correlation matrix.
        Default to None, if `analyze()` has not
        been called.
    rotation_matrix : np.array
        The rotation matrix, if a rotation
        has been performed.

    Notes
    -----
    This code was partly derived from the excellent R package
    `psych`.

    References
    ----------
    [1] https://github.com/cran/psych/blob/master/R/fa.R

    Examples
    --------
    >>> import pandas as pd
    >>> from factor_analyzer import FactorAnalyzer
    >>> df_features = pd.read_csv('test02.csv')
    >>> fa = FactorAnalyzer()
    >>> fa.analyze(df_features, 3, rotation=None)
    >>> fa.loadings
               Factor1   Factor2   Factor3
    sex      -0.129912 -0.163982  0.738235
    zygosity  0.038996 -0.046584  0.011503
    moed      0.348741 -0.614523 -0.072557
    faed      0.453180 -0.719267 -0.075465
    faminc    0.366888 -0.443773 -0.017371
    english   0.741414  0.150082  0.299775
    math      0.741675  0.161230 -0.207445
    socsci    0.829102  0.205194  0.049308
    natsci    0.760418  0.237687 -0.120686
    vocab     0.815334  0.124947  0.176397
    """

    def __init__(self,
                 log_warnings=False):

        self.log_warnings = log_warnings

        # default matrices to None
        self.phi = None
        self.structure = None

        self.corr = None
        self.loadings = None
        self.rotation_matrix = None

    @staticmethod
    def _fit_uls_objective(psi, corr_mtx, n_factors):
        """
        The objective function passed to `minimize()` for ULS.

        Parameters
        ----------
        psi : np.array
            Value passed to minimize the objective function.
        corr_mtx : np.array
            The correlation matrix.
        n_factors : int
            The number of factors to select.

        Returns
        -------
        error : float
            The scalar error calculated from the residuals
            of the loading matrix.
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
            loadings = np.dot(vectors,
                              np.diag(np.sqrt(values)))
        else:
            loadings = vectors * np.sqrt(values[0])

        # calculate the error from the loadings model
        model = sp.dot(loadings, loadings.T)

        # note that in a more recent version of the `fa()` source
        # code on GitHub, the minres objective function only sums the
        # lower triangle of the residual matrix; this could be
        # implemented here using `np.tril()` when this change is
        # merged into the stable version of `psych`.
        residual = (corr_mtx - model)**2
        error = sp.sum(residual)
        return error

    @staticmethod
    def _fit_ml_objective(psi, corr_mtx, n_factors):
        """
        The objective function passed to `minimize()` for ML.

        Parameters
        ----------
        psi : np.array
            Value passed to minimize the objective function.
        corr_mtx : np.array
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
        The ML objective is based on the `factanal()` function
        from R's `stats` package. It may generate results different
        from the `fa()` function in `psych`.

        References
        ----------
        [1] https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/factanal.R
        """
        sc = np.diag(1 / np.sqrt(psi))
        sstar = np.dot(np.dot(sc, corr_mtx), sc)

        # get the eigenvalues and eigenvectors for n_factors
        values, _ = sp.linalg.eigh(sstar)
        values = values[::-1][n_factors:]

        # calculate the error
        error = -(np.sum(np.log(values) - values) -
                  n_factors + corr_mtx.shape[0])
        return error

    @staticmethod
    def _normalize_wls(solution, corr_mtx, n_factors):
        """
        Weighted least squares normalization for loadings
        estimated using MINRES.

        Parameters
        ----------
        solution : np.array
            The solution from the L-BFGS-B optimization.
        corr_mtx : np.array
            The correlation matrix.
        n_factors : int
            The number of factors to select.

        Returns
        -------
        loadings : pd.DataFrame
            The factor loading matrix
        """
        sp.fill_diagonal(corr_mtx, 1 - solution)

        # get the eigenvalues and vectors for n_factors
        values, vectors = sp.linalg.eigh(corr_mtx)

        # sort the values and vectors in ascending order
        values = values[::-1][:n_factors]
        vectors = vectors[:, ::-1][:, :n_factors]

        # calculate loadings
        # if values are smaller than 0, set them to zero
        loadings = sp.dot(vectors, sp.diag(sp.sqrt(np.maximum(values, 0))))
        return loadings

    @staticmethod
    def _normalize_ml(solution, corr_mtx, n_factors):
        """
        Normalization for loadings estimated using ML.

        Parameters
        ----------
        solution : np.array
            The solution from the L-BFGS-B optimization.
        corr_mtx : np.array
            The correlation matrix.
        n_factors : int
            The number of factors to select.

        Returns
        -------
        loadings : pd.DataFrame
            The factor loading matrix
        """
        sc = np.diag(1 / np.sqrt(solution))
        sstar = np.dot(np.dot(sc, corr_mtx), sc)

        # get the eigenvalues for n_factors
        values, vectors = sp.linalg.eigh(sstar)

        # sort the values and vectors in ascending order
        values = values[::-1][:n_factors]
        vectors = vectors[:, ::-1][:, :n_factors]

        values = np.maximum(values - 1, 0)

        # get the loadings
        loadings = np.dot(vectors,
                          np.diag(np.sqrt(values)))

        return np.dot(np.diag(np.sqrt(solution)), loadings)

    @staticmethod
    def smc(data, sort=False):
        """
        Calculate the squared multiple correlations.
        This is equivalent to regressing each variable
        on all others and calculating the r-squared values.

        Parameters
        ----------
        data : pd.DataFrame
            The dataframe used to calculate SMC.
        sort : bool, optional
            Whether to sort the values for SMC
            before returning.
            Defaults to False.

        Returns
        -------
        smc : pd.DataFrame
            The squared multiple correlations matrix.

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import FactorAnalyzer
        >>> df_features = pd.read_csv('test02.csv')
        >>> FactorAnalyzer.smc(df_features)
                       SMC
        sex       0.212047
        zygosity  0.010857
        moed      0.385399
        faed      0.453161
        faminc    0.273753
        english   0.566065
        math      0.547790
        socsci    0.677035
        natsci    0.576016
        vocab     0.660264
        """
        corr = data.corr()
        columns = data.columns

        corr_inv = sp.linalg.inv(corr)
        smc = 1 - 1 / sp.diag(corr_inv)

        smc = pd.DataFrame(smc,
                           index=columns,
                           columns=['SMC'])
        if sort:
            smc = smc.sort_values('SMC')
        return smc

    def remove_non_numeric(self, data):
        """
        Remove non-numeric columns from data,
        as these columns cannot be used in
        factor analysis.

        Parameters
        ----------
        data : pd.DataFrame
            The dataframe from which to remove
            non-numeric columns.

        Returns
        -------
        data : pd.DataFrame
            The dataframe with non-numeric columns removed.
        """
        old_column_names = data.columns.values
        data = data.loc[:, data.applymap(sp.isreal).all() == True].copy()

        # report any non-numeric columns removed
        non_numeric_columns = set(old_column_names) - set(data.columns)
        if non_numeric_columns and self.log_warnings:
            logging.warning('The following non-numeric columns '
                            'were removed: {}.'.format(', '.join(non_numeric_columns)))
        return data

    def fit_factor_analysis(self,
                            data,
                            n_factors,
                            use_smc=True,
                            bounds=(0.005, 1),
                            method='minres'):
        """
        Fit the factor analysis model using either
        minres or ml solutions.

        Parameters
        ----------
        data : pd.DataFrame
            The data to fit.
        n_factors : int
            The number of factors to select.
        use_smc : bool
            Whether to use squared multiple correlation
            as starting guesses for factor analysis.
            Defaults to True.
        bounds : tuple
            The lower and upper bounds on the variables
            for "L-BFGS-B" optimization.
            Defaults to (0.005, 1).
        method : {'minres', 'ml'}
            The fitting method to use, either MINRES or
            Maximum Likelihood.
            Defaults to 'minres'.

        Returns
        -------
        loadings : pd.DataFrame
            The factor loadings matrix.

        Raises
        ------
        ValueError
            If any of the correlations are null, most likely due
            to having zero standard deviation.
        """
        if method not in ['ml', 'minres'] and self.log_warnings:
            logging.warning("You have selected a method other than 'minres' or 'ml'. "
                            "MINRES will be used by default, as {} is not a valid "
                            "option.".format(method))

        corr = data.corr()

        # if any variables have zero standard deviation, then
        # the correlation will be NaN, as you cannot divide by zero:
        # corr(i, j) = cov(i, j) / (stdev(i) * stdev(j))
        if corr.isnull().any().any():
            raise ValueError('The correlation matrix cannot have '
                             'features that are null or infinite. '
                             'Check to make sure you do not have any '
                             'features with zero standard deviation.')

        corr = corr.values

        # if `use_smc` is True, get get squared multiple correlations
        # and use these as initial guesses for optimizer
        if use_smc:
            smc_mtx = self.smc(data).values
            start = (np.diag(corr) - smc_mtx.T).squeeze()

        # otherwise, just start with a guess of 0.5 for everything
        else:
            start = [0.5 for _ in range(corr.shape[0])]

        # if `bounds`, set initial boundaries for all variables;
        # this must be a list passed to `minimize()`
        if bounds is not None:
            bounds = [bounds for _ in range(corr.shape[0])]

        # minimize the appropriate objective function
        # and the L-BFGS-B algorithm
        if method == 'ml':
            objective = self._fit_ml_objective
        else:
            objective = self._fit_uls_objective

        res = minimize(objective,
                       start,
                       method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 1000},
                       args=(corr, n_factors))

        if not res.success and self.log_warnings:
            logging.warning('Failed to converge: {}'.format(res.message))

        # get factor column names
        columns = ['Factor{}'.format(i) for i in range(1, n_factors + 1)]

        # transform the final loading matrix (using wls for MINRES,
        # and ml normalization for ML), and convert to DataFrame
        if method == 'ml' or method == 'mle':
            loadings = self._normalize_ml(res.x, corr, n_factors)
        else:
            loadings = self._normalize_wls(res.x, corr, n_factors)

        loadings = pd.DataFrame(loadings,
                                index=data.columns.values,
                                columns=columns)
        return loadings

    def analyze(self,
                data,
                n_factors=3,
                rotation='promax',
                method='minres',
                use_smc=True,
                bounds=(0.005, 1),
                normalize=True,
                impute='median',
                **kwargs):
        """
        Fit the factor analysis model using either
        minres or ml solutions. By default, use SMC
        as starting guesses and perform Kaiser normalization.

        Parameters
        ----------
        data : pd.DataFrame
            The data to analyze.
        n_factors : int, optional
            The number of factors to select.
            Defaults to 3.
        rotation : str, optional
            The type of rotation to perform after
            fitting the factor analysis model.
            If set to None, no rotation will be performed,
            nor will any associated Kaiser normalization.

            Methods include:

                (a) varimax (orthogonal rotation)
                (b) promax (oblique rotation)
                (c) oblimin (oblique rotation)
                (d) oblimax (orthogonal rotation)
                (e) quartimin (oblique rotation)
                (f) quartimax (orthogonal rotation)
                (g) equamax (orthogonal rotation)

            Defaults to 'promax'.

        method : {'minres', 'ml'}, optional
            The fitting method to use, either MINRES or
            Maximum Likelihood.
            Defaults to 'minres'.
        use_smc : bool, optional
            Whether to use squared multiple correlation
            as starting guesses for factor analysis.
            Defaults to True.
        bounds : tuple, optional
            The lower and upper bounds on the variables
            for "L-BFGS-B" optimization.
            Defaults to (0.005, 1).
        normalize : bool, optional
            Whether to perform Kaiser normalization
            and de-normalization prior to and following
            rotation.
            Defaults to True.
        impute : {'drop', 'mean', 'median'}, optional
            If missing values are present in the data, either use
            list-wise deletion ('drop') or impute the column median
            ('median') or column mean ('mean').
            Defaults to 'median'.
        kwargs, optional
            Additional key word arguments
            are passed to the rotation method.

        Raises
        ------
        ValueError
            If rotation not `None` or in `POSSIBLE_ROTATIONS`.
        ValueError
            If missing values present and `missing_values` is
            not set to either 'drop' or 'impute'.

        Notes
        -----
        varimax is an orthogonal rotation, while promax
        is an oblique rotation. For more details on promax
        rotations, see here:

        References
        ----------
        [1] https://www.rdocumentation.org/packages/psych/versions/1.7.8/topics/Promax
        """

        if rotation not in POSSIBLE_ROTATIONS + [None]:
            raise ValueError("The value for `rotation` must `None` or in the "
                             "set: {}.".format(', '.join(POSSIBLE_ROTATIONS)))

        df = data.copy()

        # remove non-numeric columns
        df = self.remove_non_numeric(df)

        if df.isnull().any().any():

            # impute median, if `impute` is set to 'median'
            if impute == 'median':
                df = df.apply(lambda x: x.fillna(x.median()), axis=0)

            # impute mean, if `impute` is set to 'mean'
            elif impute == 'mean':
                df = df.apply(lambda x: x.fillna(x.mean()), axis=0)

            # drop missing if `impute` is set to 'drop'
            elif impute == 'drop':
                df = df.dropna()

            else:
                raise ValueError("You have missing values in your data, but "
                                 "`impute` was not set to either 'drop', "
                                 "'mean', or 'median'.")

        # scale the data
        X = (df - df.mean(0)) / df.std(0)

        # fit factor analysis model
        loadings = self.fit_factor_analysis(X,
                                            n_factors,
                                            use_smc,
                                            bounds,
                                            method)

        # only used if we do an oblique rotations
        phi = None
        structure = None

        # default rotation matrix to None
        rotation_mtx = None

        # whether to rotate the loadings matrix
        if rotation is not None:

            if loadings.shape[1] > 1:
                rotator = Rotator()
                loadings, rotation_mtx, phi = rotator.rotate(loadings,
                                                             rotation,
                                                             normalize=normalize,
                                                             **kwargs)

                if rotation != 'promax':

                    # update the rotation matrix for everything except promax
                    rotation_mtx = np.linalg.inv(rotation_mtx).T

            else:
                warnings.warn('No rotation will be performed when '
                              'the number of factors equals 1.')

        if n_factors > 1:

            # update loading signs to match column sums
            # this is to ensure that signs align with R
            signs = np.sign(loadings.sum(0))
            signs[(signs == 0)] = 1
            loadings = pd.DataFrame(np.dot(loadings, np.diag(signs)),
                                    index=loadings.index,
                                    columns=loadings.columns)

            if phi is not None:

                # update phi, if it exists -- that is, if the rotation is oblique
                phi = np.dot(np.dot(np.diag(signs), phi), np.diag(signs))

                # create the structure matrix for any oblique rotation
                structure = np.dot(loadings, phi) if rotation in OBLIQUE_ROTATIONS else None
                structure = pd.DataFrame(structure, columns=loadings.columns, index=loadings.index)

        self.phi = phi
        self.structure = structure

        self.corr = df.corr()
        self.loadings = loadings
        self.rotation_matrix = rotation_mtx

    def get_eigenvalues(self):
        """
        Calculate the eigenvalues, given the
        factor correlation matrix.

        Return
        ------
        e_values : pd.DataFrame
            A dataframe with original eigenvalues.
        values : pd.DataFrame
            A dataframe with common-factor eigenvalues.

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import FactorAnalyzer
        >>> df_features = pd.read_csv('test02.csv')
        >>> fa = FactorAnalyzer()
        >>> fa.analyze(df_features, 3, rotation=None)
        >>> ev, v = fa.get_eigenvalues()
        >>> ev
           Original_Eigenvalues
        0              3.510189
        1              1.283710
        2              0.737395
        3              0.133471
        4              0.034456
        5              0.010292
        6             -0.007400
        7             -0.036948
        8             -0.059591
        9             -0.074281
        >>> v
           Common_Factor_Eigenvalues
        0                   3.510189
        1                   1.283710
        2                   0.737395
        3                   0.133471
        4                   0.034456
        5                   0.010292
        6                  -0.007400
        7                  -0.036948
        8                  -0.059591
        9                  -0.074281
        """
        if (self.corr is not None and self.loadings is not None):

            corr = self.corr.values

            e_values, _ = sp.linalg.eigh(corr)
            e_values = pd.DataFrame(e_values[::-1],
                                    columns=['Original_Eigenvalues'])

            communalities = self.get_communalities()
            np.fill_diagonal(corr, communalities)

            values, _ = sp.linalg.eigh(corr)
            values = pd.DataFrame(values[::-1],
                                  columns=['Common_Factor_Eigenvalues'])

            return e_values, values

    def get_communalities(self):
        """
        Calculate the communalities, given the
        factor loading matrix.

        Return
        ------
        communalities : pd.DataFrame
            A dataframe with communalities information.

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import FactorAnalyzer
        >>> df_features = pd.read_csv('test02.csv')
        >>> fa = FactorAnalyzer()
        >>> fa.analyze(df_features, 3, rotation=None)
        >>> fa.get_communalities()
                  Communalities
        sex            0.588758
        zygosity       0.003823
        moed           0.504524
        faed           0.728412
        faminc         0.331843
        english        0.662084
        math           0.619110
        socsci         0.731946
        natsci         0.649296
        vocab          0.711497
        """
        if self.loadings is not None:

            communalities = (self.loadings ** 2).sum(axis=1)
            communalities = pd.DataFrame(communalities,
                                         columns=['Communalities'])

            return communalities

    def get_uniqueness(self):
        """
        Calculate the uniquenesses, given the
        factor loading matrix.

        Return
        ------
        uniqueness : pd.DataFrame
            A dataframe with uniqueness information.

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import FactorAnalyzer
        >>> df_features = pd.read_csv('test02.csv')
        >>> fa = FactorAnalyzer()
        >>> fa.analyze(df_features, 3, rotation=None)
        >>> fa.get_uniqueness()
                  Uniqueness
        sex         0.411242
        zygosity    0.996177
        moed        0.495476
        faed        0.271588
        faminc      0.668157
        english     0.337916
        math        0.380890
        socsci      0.268054
        natsci      0.350704
        vocab       0.288503
        """
        if self.loadings is not None:

            communalities = self.get_communalities()
            uniqueness = (1 - communalities)
            uniqueness.columns = ['Uniqueness']
            return uniqueness

    def get_factor_variance(self):
        """
        Calculate the factor variance information,
        including variance, proportional variance
        and cumulative variance.

        Return
        ------
        variance_info : pd.DataFrame
            A dataframe with variance information.

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import FactorAnalyzer
        >>> df_features = pd.read_csv('test02.csv')
        >>> fa = FactorAnalyzer()
        >>> fa.analyze(df_features, 3, rotation=None)
        >>> fa.get_factor_variance()
                         Factor1   Factor2   Factor3
        SS Loadings     3.510189  1.283710  0.737395
        Proportion Var  0.351019  0.128371  0.073739
        Cumulative Var  0.351019  0.479390  0.553129
        """
        if self.loadings is not None:

            loadings = self.loadings

            n_rows = loadings.shape[0]

            # calculate variance
            loadings = loadings ** 2
            variance = loadings.sum(axis=0)

            # calculate proportional variance
            proportional_variance = variance / n_rows

            # calculate cumulative variance
            cumulative_variance = proportional_variance.cumsum(axis=0)

            # package variance info
            variance_info = pd.DataFrame([variance,
                                          proportional_variance,
                                          cumulative_variance],
                                         index=['SS Loadings',
                                                'Proportion Var',
                                                'Cumulative Var'])

            return variance_info

    def get_scores(self, data):
        """
        Get the factor scores, given the data.

        Parameters
        ----------
        data : pd.DataFrame
            The data to calculate factor scores.

        Returns
        -------
        scores : pd.DataFrame
            The factor scores.

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import FactorAnalyzer
        >>> df_features = pd.read_csv('tests/data/test02.csv')
        >>> fa = FactorAnalyzer()
        >>> fa.analyze(df_features, 3, rotation='varimax')
        >>> fa.get_scores(df_features).head()
            Factor1   Factor2   Factor3
        0 -1.158106  0.081212  0.342195
        1 -1.799933  0.155316  0.311530
        2 -0.557422 -1.596457  0.548574
        3 -0.973182 -1.530071  0.543792
        4 -1.450108 -1.553214  0.446574
        """
        if self.loadings is not None:

            df = data.copy()
            corr = data.corr()

            # scale the data
            X = (df - df.mean(0)) / df.std(0)

            if self.structure is not None:
                structure = self.structure
            else:
                structure = self.loadings

            try:
                weights = np.linalg.solve(corr, structure)
            except Exception as error:
                warnings.warn('Unable to calculate the factor score weights; '
                              'factor loadings used instead: {}'.format(error))
                weights = self.loadings

            scores = np.dot(X, weights)
            scores = pd.DataFrame(scores, columns=structure.columns)
            return scores
