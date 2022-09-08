"""
Confirmatory factor analysis using machine learning methods.

:author: Jeremy Biggs (jeremy.m.biggs@gmail.com)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: Educational Testing Service
:date: 2022-09-05
"""

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .utils import (
    commutation_matrix,
    cov,
    covariance_to_correlation,
    duplication_matrix_pre_post,
    get_free_parameter_idxs,
    get_symmetric_lower_idxs,
    get_symmetric_upper_idxs,
    impute_values,
    merge_variance_covariance,
    unique_elements,
)


class ModelSpecification:
    """
    Encapsulate the model specification for CFA.

    This class contains a number of specification properties
    that are used in the CFA procedure.

    Parameters
    ----------
    loadings : array-like
        The factor loadings specification.
    n_factors : int
        The number of factors.
    n_variables : int
        The number of variables.
    factor_names : list of str or None
        A list of factor names, if available.
        Defaults to ``None``.
    variable_names : list of str or None
        A list of variable names, if available.
        Defaults to ``None``.
    """

    def __init__(
        self, loadings, n_factors, n_variables, factor_names=None, variable_names=None
    ):
        """Initialize the specification."""
        assert isinstance(loadings, np.ndarray)
        assert loadings.shape[0] == n_variables
        assert loadings.shape[1] == n_factors

        self._loadings = loadings
        self._n_factors = n_factors
        self._n_variables = n_variables
        self._factor_names = factor_names
        self._variable_names = variable_names

        self._n_lower_diag = get_symmetric_lower_idxs(n_factors, False).shape[0]

        self._error_vars = np.full((n_variables, 1), np.nan)
        self._factor_covs = np.full((n_factors, n_factors), np.nan)

        self._loadings_free = get_free_parameter_idxs(loadings, eq=1)
        self._error_vars_free = merge_variance_covariance(self._error_vars)
        self._error_vars_free = get_free_parameter_idxs(self._error_vars_free, eq=-1)
        self._factor_covs_free = get_symmetric_lower_idxs(n_factors, False)

    def __str__(self):
        """Represent model specification as a string."""
        return f"<ModelSpecification object at {hex(id(self))}>"

    def copy(self):
        """Return a copy of the model specification."""
        return deepcopy(self)

    @property
    def loadings(self):
        """Get the factor loadings specification."""
        return self._loadings.copy()

    @property
    def error_vars(self):
        """Get the error variance specification."""
        return self._error_vars.copy()

    @property
    def factor_covs(self):
        """Get the factor covariance specification."""
        return self._factor_covs.copy()

    @property
    def loadings_free(self):
        """Get the indices of "free" factor loading parameters."""
        return self._loadings_free.copy()

    @property
    def error_vars_free(self):
        """Get the indices of "free" error variance parameters."""
        return self._error_vars_free.copy()

    @property
    def factor_covs_free(self):
        """Get the indices of "free" factor covariance parameters."""
        return self._factor_covs_free.copy()

    @property
    def n_variables(self):
        """Get the number of variables."""
        return self._n_variables

    @property
    def n_factors(self):
        """Get the number of factors."""
        return self._n_factors

    @property
    def n_lower_diag(self):
        """Get the lower diagonal of the factor covariance matrix."""
        return self._n_lower_diag

    @property
    def factor_names(self):
        """Get list of factor names, if available."""
        return self._factor_names

    @property
    def variable_names(self):
        """Get list of variable names, if available."""
        return self._variable_names

    def get_model_specification_as_dict(self):
        """
        Get the model specification as a dictionary.

        Returns
        -------
        model_specification : dict
            The model specification keys and values,
            as a dictionary.
        """
        return {
            "loadings": self._loadings.copy(),
            "error_vars": self._error_vars.copy(),
            "factor_covs": self._factor_covs.copy(),
            "loadings_free": self._loadings_free.copy(),
            "error_vars_free": self._error_vars_free.copy(),
            "factor_covs_free": self._factor_covs_free.copy(),
            "n_variables": self._n_variables,
            "n_factors": self._n_factors,
            "n_lower_diag": self._n_lower_diag,
            "variable_names": self._variable_names,
            "factor_names": self._factor_names,
        }


class ModelSpecificationParser:
    """
    Generate the model specification for CFA.

    This class includes two static methods to generate a
    :class:`ModelSpecification` object from either a dictionary
    or a numpy array.
    """

    @staticmethod
    def parse_model_specification_from_dict(X, specification=None):
        """
        Generate the model specification from a dictionary.

        The keys in the dictionary should be the factor names, and the
        values should be the feature names. If this method is used to
        create the :class:`ModelSpecification` object, then factor names
        and variable names will be added as properties to that object.

        Parameters
        ----------
        X : array-like
            The data set that will be used for CFA.
        specification : dict or None
            A dictionary with the loading details. If ``None``, the matrix will
            be created assuming all variables load on all factors.
            Defaults to ``None``.

        Returns
        -------
        ModelSpecification
            A model specification object.

        Raises
        ------
        ValueError
            If ``specification`` is not in the expected format.

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import (ConfirmatoryFactorAnalyzer,
        ...                              ModelSpecificationParser)
        >>> X = pd.read_csv('tests/data/test11.csv')
        >>> model_dict = {"F1": ["V1", "V2", "V3", "V4"],
        ...               "F2": ["V5", "V6", "V7", "V8"]}
        >>> model_spec = ModelSpecificationParser.parse_model_specification_from_dict(X, model_dict)
        """
        if specification is None:
            factor_names, variable_names = None, None
            n_variables, n_factors = X.shape[1], X.shape[1]
            loadings = np.ones((n_factors, n_factors), dtype=int)
        elif isinstance(specification, dict):
            factor_names = list(specification)
            variable_names = unique_elements(
                [v for f in specification.values() for v in f]
            )
            loadings_new = {}
            for factor in factor_names:
                loadings_for_factor = pd.Series(variable_names).isin(
                    specification[factor]
                )
                loadings_for_factor = loadings_for_factor.astype(int)
                loadings_new[factor] = loadings_for_factor
            loadings = pd.DataFrame(loadings_new).values
            n_variables, n_factors = loadings.shape
        else:
            raise ValueError(
                "The model `specification` must be either a dict "
                "or None, not {}".format(type(specification))
            )

        return ModelSpecification(
            **{
                "loadings": loadings,
                "n_variables": n_variables,
                "n_factors": n_factors,
                "factor_names": factor_names,
                "variable_names": variable_names,
            }
        )

    @staticmethod
    def parse_model_specification_from_array(X, specification=None):
        """
        Generate the model specification from a numpy array.

        The columns should correspond to the factors, and the rows
        should correspond to the variables. If this method is used
        to create the :class:`ModelSpecification` object, then *no* factor
        names and variable names will be added as properties to that
        object.

        Parameters
        ----------
        X : array-like
            The data set that will be used for CFA.
        specification : array-like or None
            An array with the loading details. If ``None``, the matrix will
            be created assuming all variables load on all factors.
            Defaults to ``None``.

        Returns
        -------
        ModelSpecification
            A model specification object.

        Raises
        ------
        ValueError
            If ``specification`` is not in the expected format.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from factor_analyzer import (ConfirmatoryFactorAnalyzer,
        ...                              ModelSpecificationParser)
        >>> X = pd.read_csv('tests/data/test11.csv')
        >>> model_array = np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]])
        >>> model_spec = ModelSpecificationParser.parse_model_specification_from_array(X,
        ...                                                                            model_array)
        """
        if specification is None:
            n_variables, n_factors = X.shape[1], X.shape[1]
            loadings = np.ones((n_factors, n_factors), dtype=int)
        elif isinstance(specification, (np.ndarray, pd.DataFrame)):
            n_variables, n_factors = specification.shape
            if isinstance(specification, pd.DataFrame):
                loadings = specification.values.copy()
            else:
                loadings = specification.copy()
        else:
            raise ValueError(
                "The model `specification` must be either a numpy array "
                "or None, not {}".format(type(specification))
            )

        return ModelSpecification(
            **{"loadings": loadings, "n_variables": n_variables, "n_factors": n_factors}
        )


class ConfirmatoryFactorAnalyzer(BaseEstimator, TransformerMixin):
    """
    Fit a confirmatory factor analysis model using maximum likelihood.

    Parameters
    ----------
    specification : :class:`ModelSpecification` or None, optional
        A model specification. This must be a :class:`ModelSpecification` object
        or ``None``. If ``None``, a :class:`ModelSpecification` object will be
        generated assuming that ``n_factors`` == ``n_variables``, and that
        all variables load on all factors. Note that this could mean the
        factor model is not identified, and the optimization could fail.
        Defaults to `None`.
    n_obs : int or None, optional
        The number of observations in the original data set.
        If this is not passed and ``is_cov_matrix`` is ``True``,
        then an error will be raised.
        Defaults to ``None``.
    is_cov_matrix : bool, optional
        Whether the input ``X`` is a covariance matrix. If ``False``,
        assume it is the full data set.
        Defaults to ``False``.
    bounds : list of tuples or None, optional
        A list of minimum and maximum boundaries for each element
        of the input array. This must equal ``x0``, which is the
        input array from your parsed and combined model specification.

        The length is:
        ((n_factors * n_variables) + n_variables + n_factors +
        (((n_factors * n_factors) - n_factors) // 2)

        If `None`, nothing will be bounded.
        Defaults to ``None``.
    max_iter : int, optional
        The maximum number of iterations for the optimization routine.
        Defaults to 200.
    tol : float or None, optional
        The tolerance for convergence.
        Defaults to ``None``.
    disp : bool, optional
        Whether to print the scipy optimization ``fmin`` message to
        standard output.
        Defaults to ``True``.

    Raises
    ------
    ValueError
        If `is_cov_matrix` is `True`, and `n_obs` is not provided.

    Attributes
    ----------
    model : ModelSpecification
        The model specification object.
    loadings_ : :obj:`numpy.ndarray`
        The factor loadings matrix.
        ``None``, if ``fit()``` has not been called.
    error_vars_ : :obj:`numpy.ndarray`
        The error variance matrix
    factor_varcovs_ : :obj:`numpy.ndarray`
        The factor covariance matrix.
    log_likelihood_ : float
        The log likelihood from the optimization routine.
    aic_ : float
        The Akaike information criterion.
    bic_ : float
        The Bayesian information criterion.

    Examples
    --------
    >>> import pandas as pd
    >>> from factor_analyzer import (ConfirmatoryFactorAnalyzer,
    ...                              ModelSpecificationParser)
    >>> X = pd.read_csv('tests/data/test11.csv')
    >>> model_dict = {"F1": ["V1", "V2", "V3", "V4"],
    ...               "F2": ["V5", "V6", "V7", "V8"]}
    >>> model_spec = ModelSpecificationParser.parse_model_specification_from_dict(X, model_dict)
    >>> cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
    >>> cfa.fit(X.values)
    >>> cfa.loadings_
    array([[0.99131285, 0.        ],
           [0.46074919, 0.        ],
           [0.3502267 , 0.        ],
           [0.58331488, 0.        ],
           [0.        , 0.98621042],
           [0.        , 0.73389239],
           [0.        , 0.37602988],
           [0.        , 0.50049507]])
    >>> cfa.factor_varcovs_
    array([[1.        , 0.17385704],
           [0.17385704, 1.        ]])
    >>> cfa.get_standard_errors()
    (array([[0.06779949, 0.        ],
           [0.04369956, 0.        ],
           [0.04153113, 0.        ],
           [0.04766645, 0.        ],
           [0.        , 0.06025341],
           [0.        , 0.04913149],
           [0.        , 0.0406604 ],
           [0.        , 0.04351208]]),
     array([0.11929873, 0.05043616, 0.04645803, 0.05803088,
            0.10176889, 0.06607524, 0.04742321, 0.05373646]))
    >>> cfa.transform(X.values)
    array([[-0.46852166, -1.08708035],
           [ 2.59025301,  1.20227783],
           [-0.47215977,  2.65697245],
           ...,
           [-1.5930886 , -0.91804114],
           [ 0.19430887,  0.88174818],
           [-0.27863554, -0.7695101 ]])
    """

    def __init__(
        self,
        specification=None,
        n_obs=None,
        is_cov_matrix=False,
        bounds=None,
        max_iter=200,
        tol=None,
        impute="median",
        disp=True,
    ):
        """Initialize the analyzer object."""
        # if the input is going to be a covariance matrix, rather than
        # the full data set, then users must pass the number of observations
        if is_cov_matrix and n_obs is None:
            raise ValueError(
                "If `is_cov_matrix=True`, you must provide "
                "the number of observations, `n_obs`."
            )

        self.specification = specification
        self.n_obs = n_obs
        self.is_cov_matrix = is_cov_matrix
        self.bounds = bounds
        self.max_iter = max_iter
        self.tol = tol
        self.impute = impute
        self.disp = disp

        self.cov_ = None
        self.mean_ = None
        self.loadings_ = None
        self.error_vars_ = None
        self.factor_varcovs_ = None

        self.log_likelihood_ = None
        self.aic_ = None
        self.bic_ = None

        self._n_factors = None
        self._n_variables = None
        self._n_lower_diag = None

    @staticmethod
    def _combine(
        loadings,
        error_vars,
        factor_vars,
        factor_covs,
        n_factors,
        n_variables,
        n_lower_diag,
    ):
        """
        Combine given multi-dimensional matrics into a one-dimensional matrix.

        Combine a set of multi-dimensional loading, error variance, factor
        variance, and factor covariance matrices into a one-dimensional
        (X, 1) matrix, where the length of X is the length of all elements
        across all of the matrices.

        Parameters
        ----------
        loadings : array-like
            The loadings matrix (``n_factors`` * ``n_variables``)
        error_vars : array-like (``n_variables`` * 1)
            The error variance array.
        factor_vars : array-like (``n_factors`` * 1)
            The factor variance array.
        factor_covs : array-like (``n_lower_diag`` * 1)
            The factor covariance array.
        n_factors : int
            The number of factors.
        n_variables : int
            The number of variables.
        n_lower_diag :int
            The number of elements in the ``factor_covs`` array, which
            is equal to the lower diagonal of the factor covariance
            matrix.

        Returns
        -------
        array : :obj:`numpy.ndarray`
            The combined (X, 1) array.
        """
        loadings = loadings.reshape(n_factors * n_variables, 1, order="F")
        error_vars = error_vars.reshape(n_variables, 1, order="F")
        factor_vars = factor_vars.reshape(n_factors, 1, order="F")
        factor_covs = factor_covs.reshape(n_lower_diag, 1, order="F")
        return np.concatenate([loadings, error_vars, factor_vars, factor_covs])

    @staticmethod
    def _split(x, n_factors, n_variables, n_lower_diag):
        """
        Split given one-dimensional array into multi-dimensional arrays.

        Given a one-dimensional array, split it into a set of
        multi-dimensional loading, error variance, factor variance,
        and factor covariance matrices.

        Parameters
        ----------
        x : array-like
            The combined (X, 1) array to split.
        n_factors : int
            The number of factors.
        n_variables : int
            The number of variables.
        n_lower_diag : int
            The number of elements in the ``factor_covs`` array, which is
            equal to the lower diagonal of the factor covariance matrix.

        Returns
        -------
        loadings : array-like
            The loadings matrix (``n_factors`` * ``n_variables``)
        error_vars : array-like (``n_variables`` * 1)
            The error variance array.
        factor_vars : array-like (``n_factors`` * 1)
            The factor variance array.
        factor_covs : array-like (``n_lower_diag`` * 1)
            The factor covariance array.
        """
        loadings_ix = int(n_factors * n_variables)
        error_vars_ix = n_variables + loadings_ix
        factor_vars_ix = n_factors + error_vars_ix
        factor_covs_ix = n_lower_diag + factor_vars_ix
        return (
            x[:loadings_ix].reshape((n_variables, n_factors), order="F"),
            x[loadings_ix:error_vars_ix].reshape((n_variables, 1), order="F"),
            x[error_vars_ix:factor_vars_ix].reshape((n_factors, 1), order="F"),
            x[factor_vars_ix:factor_covs_ix].reshape((n_lower_diag, 1), order="F"),
        )

    def _objective(self, x0, cov_mtx, loadings):
        """
        Get the objective function for the analyzer.

        Parameters
        ----------
        x0: array-like
            The combined (X, 1) array. These are the initial values for
            the ``minimize`()` function.
        cov_mtx : array-like
            The covariance matrix from the original data set.
        loadings : array-like
            The loadings matrix (``n_factors`` * ``n_variables``)
            from the model parser. This tells the objective function what
            elements should be fixed.

        Returns
        -------
        error : float
            The error from the objective function.
        """
        (
            loadings_init,
            error_vars_init,
            factor_vars_init,
            factor_covs_init,
        ) = self._split(
            x0, self.model.n_factors, self.model.n_variables, self.model.n_lower_diag
        )

        # set the loadings to zero where applicable
        loadings_init[np.where(loadings == 0)] = 0

        # combine factor variances and covariances into a single matrix
        factor_varcov_init = merge_variance_covariance(
            factor_vars_init, factor_covs_init
        )

        # make the error variance into a variance-covariance matrix
        error_varcov_init = merge_variance_covariance(error_vars_init)

        # make the factor variance-covariance matrix into a correlation matrix
        with np.errstate(all="ignore"):
            factor_varcov_init = covariance_to_correlation(factor_varcov_init)

        # calculate sigma-theta, needed for the objective function
        sigma_theta = (
            loadings_init.dot(factor_varcov_init).dot(loadings_init.T)
            + error_varcov_init
        )

        with np.errstate(all="ignore"):
            error = -(
                ((-self.n_obs * self.model.n_variables / 2) * np.log(2 * np.pi))
                - (self.n_obs / 2)
                * (
                    np.log(np.linalg.det(sigma_theta))
                    + np.trace(cov_mtx.dot(np.linalg.inv(sigma_theta)))
                )
            )

            # make sure the error is greater than or
            # equal to zero before we return it; we
            # do not do this for the Bollen approach
            error = 0.0 if error < 0.0 else error

        return error

    def fit(self, X, y=None):
        """
        Perform confirmatory factor analysis.

        Parameters
        ----------
        X : array-like
            The data to use for confirmatory factor analysis. If this is just a
            covariance matrix, make sure ``is_cov_matrix`` was set to ``True``.
        y : ignored

        Raises
        ------
        ValueError
            If the specification is not None or a :class:`ModelSpecification` object.
        AssertionError
            If ``is_cov_matrix`` was ``True`` and the matrix is not square.
        AssertionError
            If ``len(bounds)`` != ``len(x0)``

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import (ConfirmatoryFactorAnalyzer,
        ...                              ModelSpecificationParser)
        >>> X = pd.read_csv('tests/data/test11.csv')
        >>> model_dict = {"F1": ["V1", "V2", "V3", "V4"],
        ...               "F2": ["V5", "V6", "V7", "V8"]}
        >>> model_spec = ModelSpecificationParser.parse_model_specification_from_dict(X, model_dict)
        >>> cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
        >>> cfa.fit(X.values)
        >>> cfa.loadings_
        array([[0.99131285, 0.        ],
               [0.46074919, 0.        ],
               [0.3502267 , 0.        ],
               [0.58331488, 0.        ],
               [0.        , 0.98621042],
               [0.        , 0.73389239],
               [0.        , 0.37602988],
               [0.        , 0.50049507]])
        """
        if self.specification is None:
            self.model = ModelSpecificationParser.parse_model_specification_from_array(
                X
            )
        elif isinstance(self.specification, ModelSpecification):
            self.model = self.specification.copy()
        else:
            raise ValueError(
                "The `specification` must be None or `ModelSpecification` "
                "instance, not {}".format(type(self.specification))
            )

        if isinstance(X, pd.DataFrame):
            X = X.values

        # now check the array, and make sure it
        # meets all of our expected criteria
        X = check_array(X, force_all_finite="allow-nan", estimator=self, copy=True)

        # check to see if there are any null values, and if
        # so impute using the desired imputation approach
        if np.isnan(X).any() and not self.is_cov_matrix:
            X = impute_values(X, how=self.impute)

        if not self.is_cov_matrix:
            # make sure that the columns are in the proper order
            # data = data[variable_names].copy()
            # get the number of observations from the data, if `n_obs` not passed;
            # then, calculate the covariance matrix from the data set
            self.n_obs = X.shape[0] if self.n_obs is None else self.n_obs
            self.mean_ = np.mean(X, axis=0)
            cov_mtx = cov(X)
        else:
            error_msg = (
                "If `is_cov_matrix=True`, then the rows and column in the data "
                "set must be equal, and must equal the number of variables "
                "in your model."
            )
            assert X.shape[0] == X.shape[1] == self.model.n_variables, error_msg
            cov_mtx = X.copy()

        self.cov_ = cov_mtx.copy()

        # we initialize all of the arrays, setting the covariances
        # lower than the expected variances, and the loadings to 1 or 0
        loading_init = self.model.loadings
        error_vars_init = np.full((self.model.n_variables, 1), 0.5)
        factor_vars_init = np.full((self.model.n_factors, 1), 1.0)
        factor_covs_init = np.full((self.model.n_lower_diag, 1), 0.05)

        # we merge all of the arrays into a single 1d vector
        x0 = self._combine(
            loading_init,
            error_vars_init,
            factor_vars_init,
            factor_covs_init,
            self.model.n_factors,
            self.model.n_variables,
            self.model.n_lower_diag,
        )

        # if the bounds argument is None, then we initialized the
        # boundaries to (None, None) for everything except factor covariances;
        # at some point in the future, we may update this to place limits
        # on the loading matrix boundaries, too, but the case in R and SAS
        if self.bounds is not None:
            error_msg = (
                "The length of `bounds` must equal the length of your "
                "input array `x0`: {} != {}.".format(len(self.bounds), len(x0))
            )
            assert len(self.bounds) == len(x0), error_msg

        # fit the actual model using L-BFGS algorithm;
        # the constraints are set inside the objective function,
        # so that we can avoid using linear programming methods (e.g. SLSQP)
        res = minimize(
            self._objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "disp": self.disp},
            bounds=self.bounds,
            args=(cov_mtx, self.model.loadings),
        )

        # if the optimizer failed to converge, print the message
        if not res.success:
            warnings.warn(
                f"The optimization routine failed to converge: {str(res.message)}"
            )

        # we split all the 1d array back into the set of original arrays
        (loadings_res, error_vars_res, factor_vars_res, factor_covs_res) = self._split(
            res.x, self.model.n_factors, self.model.n_variables, self.model.n_lower_diag
        )

        # we combine the factor covariances and variances into
        # a single variance-covariance matrix to make things easier,
        # but also check to make see if anything was fixed
        factor_varcovs_res = merge_variance_covariance(factor_vars_res, factor_covs_res)
        with np.errstate(all="ignore"):
            factor_varcovs_res = covariance_to_correlation(factor_varcovs_res)

        self.loadings_ = loadings_res
        self.error_vars_ = error_vars_res
        self.factor_varcovs_ = factor_varcovs_res

        # we also calculate the log-likelihood, AIC, and BIC
        self.log_likelihood_ = -res.fun
        self.aic_ = 2 * res.fun + 2 * (x0.shape[0] + self.model.n_variables)
        if self.n_obs is not None:
            self.bic_ = 2 * res.fun + np.log(self.n_obs) * (
                x0.shape[0] + self.model.n_variables
            )
        return self

    def transform(self, X):
        """
        Get the factor scores for a new data set.

        Parameters
        ----------
        X : array-like, shape (``n_samples``, ``n_features``)
            The data to score using the fitted factor model.

        Returns
        -------
        scores : numpy array, shape (``n_samples``, ``n_components``)
            The latent variables of X.

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import (ConfirmatoryFactorAnalyzer,
        ...                              ModelSpecificationParser)
        >>> X = pd.read_csv('tests/data/test11.csv')
        >>> model_dict = {"F1": ["V1", "V2", "V3", "V4"],
        ...               "F2": ["V5", "V6", "V7", "V8"]}
        >>> model_spec = ModelSpecificationParser.parse_model_specification_from_dict(X, model_dict)
        >>> cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
        >>> cfa.fit(X.values)
        >>> cfa.transform(X.values)
        array([[-0.46852166, -1.08708035],
               [ 2.59025301,  1.20227783],
               [-0.47215977,  2.65697245],
               ...,
               [-1.5930886 , -0.91804114],
               [ 0.19430887,  0.88174818],
           [-0.27863554, -0.7695101 ]])

        References
        ----------
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6157408/
        """
        # check if the data is a data frame,
        # so we can convert it to an array
        if isinstance(X, pd.DataFrame):
            X = X.values

        # now check the array, and make sure it
        # meets all of our expected criteria
        X = check_array(X, force_all_finite=True, estimator=self, copy=True)

        # meets all of our expected criteria
        check_is_fitted(self, ["loadings_", "error_vars_"])

        # see if we saved the original mean and std
        if self.mean_ is None:
            warnings.warn(
                "Could not find original mean; using the mean "
                "from the current data set."
            )
            mean = np.mean(X, axis=0)
        else:
            mean = self.mean_

        # get the scaled data
        X_scale = X - mean

        # get the loadings and error variances
        loadings = self.loadings_.copy()
        error_vars = self.error_vars_.copy()

        # make the error variance an identity matrix,
        # and take the inverse of this matrix
        error_covs = np.eye(error_vars.shape[0])
        np.fill_diagonal(error_covs, error_vars)
        error_covs_inv = np.linalg.inv(error_covs)

        # calculate the weights, using Bartlett's formula
        # (lambda' error_covsˆ−1 lambda)ˆ−1 lambda' error_covsˆ−1 (X - muX)
        weights = (
            np.linalg.pinv(loadings.T.dot(error_covs_inv).dot(loadings))
            .dot(loadings.T)
            .dot(error_covs_inv)
        )

        scores = weights.dot(X_scale.T).T
        return scores

    def get_model_implied_cov(self):
        """
        Get the model-implied covariance matrix (sigma) for an estimated model.

        Returns
        -------
        model_implied_cov : :obj:`numpy.ndarray`
            The model-implied covariance matrix.

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import (ConfirmatoryFactorAnalyzer,
        ...                              ModelSpecificationParser)
        >>> X = pd.read_csv('tests/data/test11.csv')
        >>> model_dict = {"F1": ["V1", "V2", "V3", "V4"],
        ...               "F2": ["V5", "V6", "V7", "V8"]}
        >>> model_spec = ModelSpecificationParser.parse_model_specification_from_dict(X, model_dict)
        >>> cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
        >>> cfa.fit(X.values)
        >>> cfa.get_model_implied_cov()
        array([[2.07938612, 0.45674659, 0.34718423, 0.57824753, 0.16997013,
                0.12648394, 0.06480751, 0.08625868],
               [0.45674659, 1.16703337, 0.16136667, 0.26876186, 0.07899988,
                0.05878807, 0.03012168, 0.0400919 ],
               [0.34718423, 0.16136667, 1.07364855, 0.20429245, 0.06004974,
                0.04468625, 0.02289622, 0.03047483],
               [0.57824753, 0.26876186, 0.20429245, 1.28809317, 0.10001495,
                0.07442652, 0.03813447, 0.05075691],
               [0.16997013, 0.07899988, 0.06004974, 0.10001495, 2.0364391 ,
                0.72377232, 0.37084458, 0.49359346],
               [0.12648394, 0.05878807, 0.04468625, 0.07442652, 0.72377232,
                1.48080077, 0.27596546, 0.36730952],
               [0.06480751, 0.03012168, 0.02289622, 0.03813447, 0.37084458,
                0.27596546, 1.11761918, 0.1882011 ],
               [0.08625868, 0.0400919 , 0.03047483, 0.05075691, 0.49359346,
                0.36730952, 0.1882011 , 1.28888233]])
        """
        # meets all of our expected criteria
        check_is_fitted(self, ["loadings_", "factor_varcovs_"])
        error = np.diag(self.error_vars_.flatten())
        return self.loadings_.dot(self.factor_varcovs_).dot(self.loadings_.T) + error

    def _get_derivatives_implied_cov(self):
        """
        Compute the derivatives for the implied covariance matrix (sigma).

        Returns
        -------
        loadings_dx : :obj:`numpy.ndarray`
            The derivative of the loadings matrix.
        factor_covs_dx : :obj:`numpy.ndarray`
            The derivative of the factor covariance matrix.
        error_covs_dx : :obj:`numpy.ndarray`
            The derivative of the error covariance matrix.
        """
        # meets all of our expected criteria
        check_is_fitted(self, "loadings_")

        loadings = self.loadings_.copy()
        factor_covs = self.factor_varcovs_.copy()

        sym_lower_var_idx = get_symmetric_lower_idxs(self.model.n_variables)
        sym_upper_fac_idx = get_symmetric_upper_idxs(self.model.n_factors, diag=False)
        sym_lower_fac_idx = get_symmetric_lower_idxs(self.model.n_factors, diag=False)

        factors_diag = np.eye(self.model.n_factors)
        factors_diag_mult = (
            factors_diag.dot(factor_covs).dot(factors_diag.T).dot(loadings.T)
        )

        # calculate the derivative of the loadings matrix, using the commutation matrix
        loadings_dx = np.eye(self.model.n_variables**2) + commutation_matrix(
            self.model.n_variables, self.model.n_variables
        )
        loadings_dx = loadings_dx.dot(
            np.kron(factors_diag_mult, np.eye(self.model.n_variables)).T
        )

        # calculate the derivative of the factor_covs matrix
        factor_covs_dx = loadings.dot(factors_diag)
        factor_covs_dx = np.kron(factor_covs_dx, factor_covs_dx)

        off_diag = (
            factor_covs_dx[:, sym_lower_fac_idx] + factor_covs_dx[:, sym_upper_fac_idx]
        )

        combine_indices = np.concatenate([sym_upper_fac_idx, sym_lower_fac_idx])
        combine_diag = np.concatenate([off_diag, off_diag], axis=1)

        factor_covs_dx[:, combine_indices] = combine_diag
        factor_covs_dx = factor_covs_dx[:, : factor_covs.size]

        # calculate the derivative of the error_cov matrix,
        # which we assume will always be a diagonal matrix
        error_covs_dx = np.eye(self.model.n_variables**2)

        # make sure these matrices are symmetric
        loadings_dx = loadings_dx[sym_lower_var_idx, :]
        factor_covs_dx = factor_covs_dx[sym_lower_var_idx, :]
        error_covs_dx = error_covs_dx[sym_lower_var_idx, :]

        # we also want to calculate the derivative for the intercepts
        intercept_dx = np.zeros(
            (loadings_dx.shape[0], self.model.n_variables), dtype=float
        )
        return (
            loadings_dx[:, self.model.loadings_free].copy(),
            factor_covs_dx[:, self.model.factor_covs_free].copy(),
            error_covs_dx[:, self.model.error_vars_free].copy(),
            intercept_dx,
        )

    def _get_derivatives_implied_mu(self):
        """
        Compute the "derivatives" for the implied means.

        Note that the derivatives of the implied means will not correspond
        to the actual mean values of the original data set, because that
        data could be a covariance matrix, rather than the full data set.
        Thus, we assume the mean values are zero and the data are normally
        distributed.

        Returns
        -------
        loadings_dx : :obj:`numpy.ndarray`
            The derivative of the loadings means.
        factor_covs_dx : :obj:`numpy.ndarray`
            The derivative of the factor covariance means.
        error_covs_dx : :obj:`numpy.ndarray`
            The derivative of the error covariance means.
        """
        # meets all of our expected criteria
        check_is_fitted(self, "loadings_")

        # initialize some matrices that we'll use below to
        # correct the shape of the mean loadings derivatives
        factors_zero = np.zeros((self.model.n_factors, 1))
        factors_diag = np.eye(self.model.n_factors)

        # the mean derivatives will just be zeros for both
        # the error covariance and factor covariance matrices,
        # since we don't have actual mean values
        error_covs_dx = np.zeros(
            (self.model.n_variables, len(self.model.error_vars_free))
        )
        factor_covs_dx = np.zeros(
            (self.model.n_variables, len(self.model.factor_covs_free))
        )

        # again, the implied means are just going to be diagonal matrices
        # corresponding to the number of variables and factors
        loadings_dx = np.kron(
            factors_diag.dot(factors_zero).T, np.eye(self.model.n_variables)
        )
        loadings_dx = loadings_dx[:, self.model.loadings_free].copy()

        # we also calculate the derivative for the intercept, which will be zeros again
        intercept_dx = np.zeros((loadings_dx.shape[0], self.model.n_variables))
        intercept_dx[: self.model.n_variables, : self.model.n_variables] = np.eye(
            self.model.n_variables
        )

        return (loadings_dx, factor_covs_dx, error_covs_dx, intercept_dx)

    def get_standard_errors(self):
        """
        Get standard errors from the implied covariance matrix and implied means.

        Returns
        -------
        loadings_se : :obj:`numpy.ndarray`
            The standard errors for the factor loadings.
        error_vars_se : :obj:`numpy.ndarray`
            The standard errors for the error variances.

        Examples
        --------
        >>> import pandas as pd
        >>> from factor_analyzer import (ConfirmatoryFactorAnalyzer,
        ...                              ModelSpecificationParser)
        >>> X = pd.read_csv('tests/data/test11.csv')
        >>> model_dict = {"F1": ["V1", "V2", "V3", "V4"],
        ...               "F2": ["V5", "V6", "V7", "V8"]}
        >>> model_spec = ModelSpecificationParser.parse_model_specification_from_dict(X, model_dict)
        >>> cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
        >>> cfa.fit(X.values)
        >>> cfa.get_standard_errors()
        (array([[0.06779949, 0.        ],
               [0.04369956, 0.        ],
               [0.04153113, 0.        ],
               [0.04766645, 0.        ],
               [0.        , 0.06025341],
               [0.        , 0.04913149],
               [0.        , 0.0406604 ],
               [0.        , 0.04351208]]),
         array([0.11929873, 0.05043616, 0.04645803, 0.05803088,
                0.10176889, 0.06607524, 0.04742321, 0.05373646]))
        """
        # meets all of our expected criteria
        check_is_fitted(self, ["loadings_", "n_obs"])

        (
            loadings_dx,
            factor_covs_dx,
            error_covs_dx,
            intercept_dx,
        ) = self._get_derivatives_implied_cov()

        (
            loadings_dx_mu,
            factor_covs_dx_mu,
            error_covs_dx_mu,
            intercept_dx_mu,
        ) = self._get_derivatives_implied_mu()

        # combine all of our derivatives; below we will  merge all of these
        # together in a single matrix, delta, to use the delta rule
        # (basically, using the gradients to calculate the information)
        loadings_dx = np.append(loadings_dx_mu, loadings_dx, axis=0)
        factor_covs_dx = np.append(factor_covs_dx_mu, factor_covs_dx, axis=0)
        error_cov_dx = np.append(error_covs_dx_mu, error_covs_dx, axis=0)
        intercept_dx = np.append(intercept_dx_mu, intercept_dx, axis=0)

        # get get the implied covariance, invert it, and take the Kronecker product
        sigma = self.get_model_implied_cov()
        sigma_inv = np.linalg.inv(sigma)
        sigma_inv_kron = np.kron(sigma_inv, sigma_inv)

        # we get the fisher information matrix for H1, which is the unrestricted
        # model information; we'll use this with the deltas to calculate the full
        # (inverted) information matrix below, and then invert the whole thing
        h1_information = 0.5 * duplication_matrix_pre_post(sigma_inv_kron)
        h1_information = block_diag(sigma_inv, h1_information)

        # we concatenate all derivatives into a delta matrix
        delta = np.concatenate(
            (loadings_dx, error_cov_dx, factor_covs_dx, intercept_dx), axis=1
        )

        # calculate the fisher information matrix
        information = delta.T.dot(h1_information).dot(delta)
        information = (1 / self.n_obs) * np.linalg.inv(information)

        # calculate the standard errors from the diagonal of the
        # information / cov matrix; also take the absolute value,
        # just in case anything is negative
        se = np.sqrt(np.abs(np.diag(information)))

        # get the indexes for the standard errors
        # for the loadings and the errors variances;
        # in the future, we may add the factor and intercept
        # covariances, but these sometimes require transformations
        # that are more complicated, so for now we won't return them
        loadings_idx = len(self.model.loadings_free)
        error_vars_idx = self.model.n_variables + loadings_idx

        # get the loading standard errors and reshape them into the
        # format of the original loadings matrix
        loadings_se = np.zeros((self.model.n_factors * self.model.n_variables,))
        loadings_se[self.model.loadings_free] = se[:loadings_idx]
        loadings_se = loadings_se.reshape(
            (self.model.n_variables, self.model.n_factors), order="F"
        )

        # get the error variance standard errors
        error_vars_se = se[loadings_idx:error_vars_idx]
        return loadings_se, error_vars_se
