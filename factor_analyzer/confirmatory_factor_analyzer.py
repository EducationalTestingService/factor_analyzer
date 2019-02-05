"""
Confirmatory factor analysis using ML.

:author: Jeremy Biggs (jbiggs@ets.org)
:date: 2/05/2019
:organization: ETS
"""

import pandas as pd
import numpy as np

from scipy.optimize import minimize


def fill_lower_diag(x):
    """
    Fill the lower diagonal of a square matrix,
    given a 1d input array.

    Parameters
    ----------
    x : np.array
        The input matrix that will be used to fill
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
    x = np.squeeze(x, axis=1)
    n = int(np.sqrt(len(x) * 2)) + 1
    out = np.zeros((n, n), dtype=float)
    out[np.tri(n, dtype=bool, k=-1)] = x
    return out


class ModelParser:
    """
    This is a class to parse the confirmatory
    factor analysis model into a format. usable
    by `ConfirmatoryFactorAnalyzer`.

    The model specifies the factor-variable relationships,
    as well as the variances and covariances of the factors
    and the error variances of the observed variables.

    Examples
    --------
    >>> from factor_analyzer import ModelParser
    >>> model = {'loadings': {'F1': ['X1', 'X2', 'X3'], 'F2': ['X4', 'X5', 'X6']},
                 'factor_covs': [0.05],
                 'factor_vars': [0.5, 0.5],
                 'error_vars': [1] * 6}
    >>> ModelParser().parse(model)
        (array([[1, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1]]),
         array([1, 1, 1, 1, 1, 1]),
         array([0.5, 0.5]),
         array([0.05]),
         ['X1', 'X2', 'X3', 'X4', 'X5', 'X6'],
         ['F1', 'F2'],
         2,
         6,
         1)

    >>> from factor_analyzer import ModelParser
    >>> model = {'loadings': {'F1': ['X1', 'X2', 'X3'], 'F2': ['X4', 'X5', 'X6']}}
    >>> ModelParser().parse(model)
        (array([[1, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1]]),
         array([nan, nan, nan, nan, nan, nan]),
         array([nan, nan]),
         array([nan]),
         ['X1', 'X2', 'X3', 'X4', 'X5', 'X6'],
         ['F1', 'F2'],
         2,
         6,
         1)

    >>> import pandas as pd
    >>> from factor_analyzer import ModelParser
    >>> loadings = pd.DataFrame({'F1': [1, 1, 1, 0, 0, 0], 'F2': [0, 0, 0, 1, 1, 1],
                                index='X1', 'X2', 'X3', 'X4', 'X5', 'X6']})
    >>> model = {'loadings': loadings}
    >>> ModelParser().parse(model)
        (array([[1, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1]]),
         array([nan, nan, nan, nan, nan, nan]),
         array([nan, nan]),
         array([nan]),
         ['X1', 'X2', 'X3', 'X4', 'X5', 'X6'],
         ['F1', 'F2'],
         2,
         6,
         1)
    """

    @staticmethod
    def parse_loadings(loadings):
        """
        Parse the loadings pattern from the model.

        Parameters
        ----------
        loadings : pd.DataFrame or dict
            The loadings pattern from the model.

        Returns
        -------
        loadings : pd.DataFrame
            The updated loadings pattern matrix.
        variable_names : list
            The names of the variables.
        factor_names : list
            The names of the factors.
        n_variables : int
            The number of variables.
        n_factors : int
            The number of factors.

        Raises
        ------
        TypeError
            If `loadings` is not a pandas
            DataFrame or a dict object.
        """
        if isinstance(loadings, pd.DataFrame):
            factor_names = loadings.columns.tolist()
            variable_names = loadings.index.tolist()
            loadings = loadings.values

        elif isinstance(loadings, dict):
            factor_names = list(loadings.keys())
            variable_names = [v for f in loadings.values() for v in f]

            loadings_new = {}
            for factor in factor_names:
                loadings_for_factor = pd.Series(variable_names).isin(loadings[factor])
                loadings_for_factor = loadings_for_factor.astype(int)
                loadings_new[factor] = loadings_for_factor

            loadings = pd.DataFrame(loadings_new).values

        else:
            raise TypeError('The `loadings` matrix must be  either '
                            'a pandas DataFrame or dict object, not '
                            '{}.'.format(type(loadings)))

        n_factors = len(factor_names)
        n_variables = len(variable_names)

        return (loadings,
                variable_names,
                factor_names,
                n_variables,
                n_factors)

    @staticmethod
    def parse_error_vars(error_vars, n_variables):
        """
        Parse the error variance pattern from the model.

        Parameters
        ----------
        error_vars : pd.DataFrame or np.array or None
            The error variance pattern from the model.
        n_variables : int
            The number of variables.

        Returns
        -------
        error_vars : np.array
            The updated error variance pattern matrix.

        Raises
        ------
        AssertionError
            If the length of `error_vars` is not
            equal to `n_variables`.
        """
        if error_vars is not None:
            error_vars = np.array(error_vars, dtype=float)
            error_msg = ('The length of `error_vars` must equal '
                         'the number of variables in your data set '
                         '{} != {}'.format(len(error_vars), n_variables))
            assert len(error_vars) == n_variables, error_msg
        else:
            error_vars = np.full((n_variables, 1), np.nan)

        return error_vars

    @staticmethod
    def parse_factor_vars(factor_vars, n_factors):
        """
        Parse the factor variance pattern from the model.

        Parameters
        ----------
        factor_vars : pd.DataFrame or np.array or None
            The factor variance pattern from the model.
        n_factors : int
            The number of factors.

        Returns
        -------
        factor_vars : np.array
            The updated factor variance pattern matrix.

        Raises
        ------
        AssertionError
            If the length of `factor_vars` is not
            equal to `n_factors`.
        """
        if factor_vars is not None:
            factor_vars = np.array(factor_vars, dtype=float)
            error_msg = ('The length of `factor_vars` must equal '
                         'the number of factors in your loading matrix '
                         '{} != {}'.format(len(factor_vars), n_factors))
            assert len(factor_vars) == n_factors, error_msg
        else:
            factor_vars = np.full((n_factors, 1), np.nan)

        return factor_vars

    @staticmethod
    def parse_factor_covs(factor_covs, n_factors):
        """
        Parse the factor covariance pattern from the model.

        Parameters
        ----------
        factor_covs : pd.DataFrame or np.array or None
            The factor covariance pattern from the model.
        n_factors : int
            The number of factors.

        Returns
        -------
        factor_covs : np.array
            The updated factor covariance pattern matrix.
        n_lower_diag : int
            The number of elements in the lower diagonal.

        Raises
        ------
        AssertionError
            If the length of `factor_covs` is not
            equal to `n_lower_diag`.
        """
        n_lower_diag = np.tril(np.ones((n_factors, n_factors)), -1)
        n_lower_diag = n_lower_diag[n_lower_diag != 0].flatten().shape[0]
        if factor_covs is not None:
            factor_covs = np.array(factor_covs, dtype=float)

            if len(factor_covs.shape) > 1:
                (factor_covs_rows,
                 factor_covs_cols) = factor_covs.shape

                if factor_covs_rows == factor_covs_cols:
                    factor_covs = np.tril(factor_covs, -1)
                    factor_covs = factor_covs[factor_covs != 0].flatten()

            error_msg = ('The length of `factor_covs` must equal '
                         'the lower diagonal of the factor covariance matrix '
                         '{} != {}'.format(len(factor_covs), n_lower_diag))
            assert len(factor_covs) == n_lower_diag, error_msg

        else:
            factor_covs = np.full((n_lower_diag, 1), np.nan)

        return factor_covs, n_lower_diag

    def parse(self, model):
        """
        Parse the model into a set of
        arrays, variable and factor names,
        and factor, variable, and lower diagonal
        dimensions.

        Parameters
        ----------
        model : dict
            A dictionary specifying the model
            inputs. In general, the `model`
            can have the following keys:
                - `loadings` : dict or pd.DataFrame
                    A dictionary where the keys are factor names
                    and the values are lists of variables that load
                    on each of those factors. Alternatively, you can
                    specify a data frame, where the columns are factor
                    names, the index is variable names, and the values
                    are 0s and 1s that indicates whether a variable
                    loads on a particular factor.
                - factor_covs : list or None, optional
                    A list of factor covariances. If None, no
                    covariance constraints will be imposed on the
                    confirmatory factor analysis model. This list
                    should be in the following format:
                        ```
                        [cov(x1, x2'), cov(x1, x3), cov(x2, x3), ...]
                        ```
                - factor_vars : list or None, optional
                    A list of factor variances. If None, no
                    factor variance constraints will be imposed on the
                    confirmatory factor analysis model.
                - error_vars : list or None, optional
                    A list of error variances. If None, no
                    error variance constraints will be imposed on the
                    confirmatory factor analysis model.

        Returns
        -------
        loadings : np.array
            The updated loadings pattern matrix.
        error_vars : np.array
            The updated error variance pattern matrix.
        factor_vars : np.array
            The updated factor variance pattern matrix.
        factor_covs : np.array
            The updated factor covariance pattern matrix.
        variable_names : list
            The names of the variables.
        factor_names : list
            The names of the factors.
        n_variables : int
            The number of variables.
        n_factors : int
            The number of factors.
        n_lower_diag : int
            The number of elements in the lower diagonal.
        """
        loadings = model['loadings']
        error_vars = model.get('errors_vars')
        factor_vars = model.get('factor_vars')
        factor_covs = model.get('factor_covs')

        (loadings,
         variable_names,
         factor_names,
         n_variables,
         n_factors) = self.parse_loadings(loadings)

        error_vars = self.parse_error_vars(error_vars, n_variables)
        factor_vars = self.parse_factor_vars(factor_vars, n_factors)

        (factor_covs,
         n_lower_diag) = self.parse_factor_covs(factor_covs, n_factors)

        return (loadings,
                error_vars,
                factor_vars,
                factor_covs,
                variable_names,
                factor_names,
                n_factors,
                n_variables,
                n_lower_diag)


class ConfirmatoryFactorAnalyzer:
    """
    A ConfirmatoryFactorAnalyzer class, which fits a
    confirmatory factor analysis model using maximum likelihood.


    Attributes
    ----------
    loadings : pd.DataFrame

    error_vars : pd.DataFrame

    factor_covs : pd.DataFrame


    Examples
    --------
    >>> import pandas as pd
    >>> from factor_analyzer import ConfirmatoryFactorAnalyzer
    >>> data = pd.read_csv('test10.csv')
    >>> model = {'loadings': {'F1': ['X1', 'X2', 'X3'], 'F2': ['X4', 'X5', 'X6']}}
    >>> cfa = ConfirmatoryFactorAnalyzer()
    >>> cfa.analyze(data, model)
    >>> cfa.loadings

    """

    def __init__(self):

        self.loadings = None
        self.error_vars = None
        self.factor_covs = None

    @staticmethod
    def combine(loadings,
                error_vars,
                factor_vars,
                factor_covs,
                n_factors,
                n_variables,
                n_lower_diag):
        """
        Combine a set of multi-dimensional loading,
        error variance, factor variance, and factor
        covariance matrices into a one-dimensional
        (X, 1) matrix, where the length of X is the
        length of all elements across all of the matrices.

        Parameters
        ----------
        loadings : array-like
            The loadings matrix (n_factors * n_variables)
        error_vars : array-like (n_variables * 1)
            The error variance array.
        factor_vars : array-like (n_factors * 1)
            The factor variance array.
        factor_covs : array-like (n_lower_diag * 1)
            The factor covariance array
        n_factors : int
            The number of factors.
        n_variables : int
            The number of variables.
        n_lower_diag :int
            The number of elements in the `factor_covs`
            array, which is equal to the lower diagonal
            of the factor covariance matrix.

        Returns
        -------
        array : np.arrays
            The combined (X, 1) array.
        """
        loadings = np.array(loadings).reshape(n_factors * n_variables, 1)
        error_vars = np.array(error_vars).reshape(n_variables, 1)
        factor_vars = np.array(factor_vars).reshape(n_factors, 1)
        factor_covs = np.array(factor_covs).reshape(n_lower_diag, 1)
        return np.concatenate([loadings,
                               error_vars,
                               factor_vars,
                               factor_covs])

    @staticmethod
    def split(x, n_factors, n_variables, n_lower_diag):
        """
        Given a one-dimensional array, split it
        into a set of multi-dimensional loading,
        error variance, factor variance, and factor
        covariance matrices.

        Parameters
        ----------
        x : np.array
            The combined (X, 1) array to split.
        n_factors : int
            The number of factors.
        n_variables : int
            The number of variables.
        n_lower_diag :int
            The number of elements in the `factor_covs`
            array, which is equal to the lower.

        Returns
        -------
        loadings : array-like
            The loadings matrix (n_factors * n_variables)
        error_vars : array-like (n_variables * 1)
            The error variance array.
        factor_vars : array-like (n_factors * 1)
            The factor variance array.
        factor_covs : array-like (n_lower_diag * 1)
            The factor covariance array
        """
        x = np.array(x)
        loadings_ix = int(n_factors * n_variables)
        error_vars_ix = n_variables + loadings_ix
        factor_vars_ix = n_factors + error_vars_ix
        factor_covs_ix = n_lower_diag + factor_vars_ix
        return (x[:loadings_ix].reshape((n_variables, n_factors)),
                x[loadings_ix:error_vars_ix].reshape((n_variables, 1)),
                x[error_vars_ix:factor_vars_ix].reshape((n_factors, 1)),
                x[factor_vars_ix:factor_covs_ix].reshape((n_lower_diag, 1)))

    def objective(self,
                  x0,
                  cov,
                  loadings,
                  error_vars,
                  factor_vars,
                  factor_covs,
                  n_factors,
                  n_variables,
                  n_lower_diag,
                  fix_first):
        """
        The objective function.

        Parameters
        ----------
        x0: np.array
            The combined (X, 1) array. These are the
            initial values for the `minimize()` function.
        cov : np.array or pd.DataFrame
            The covariance matrix from the original data
            set.
        loadings : array-like
            The loadings matrix (n_factors * n_variables)
            from the model parser. This tells the objective
            function what elements should be fixed.
        error_vars : array-like (n_variables * 1)
            The error variance array. This tells the objective
            function what elements should be fixed.
        factor_vars : array-like (n_factors * 1)
            The factor variance array. This tells the objective
            function what elements should be fixed.
        factor_covs : array-like (n_lower_diag * 1)
            The factor covariance array. This tells the objective
            function what elements should be fixed.
        n_factors : int
            The number of factors.
        n_variables : int
            The number of variables.
        n_lower_diag :int
            The number of elements in the `factor_covs`
            array, which is equal to the lower diagonal
            of the factor covariance matrix.
        fix_first : bool
            Whether to fix the first variable loading
            on a given factor to 1.

        Returns
        -------
        error : float
            The error from the objective function.
        """
        (loadings_init,
         error_vars_init,
         factor_vars_init,
         factor_covs_init) = self.split(x0, n_factors, n_variables, n_lower_diag)

        if fix_first:
            row_idx = [np.where(loadings[:, i] == 1)[0][0] for i in range(n_factors)]
            col_idx = [i for i in range(n_factors)]
            loadings_init[row_idx, col_idx] = 1

        # loadings
        loadings_init[np.where(loadings == 0)] = 0

        # factor variances
        # factor_vars_init[~pd.isna(factor_vars)] = factor_vars[~pd.isna(factor_vars)]

        # factor covariances
        # factor_covs_init[~pd.isna(factor_covs)] = factor_covs[~pd.isna(factor_covs)]

        # combine factor variances and covariances
        factor_cov_init_final = fill_lower_diag(factor_covs_init)
        factor_cov_init_final += factor_cov_init_final.T
        np.fill_diagonal(factor_cov_init_final, factor_vars_init)

        # error covarianes
        # error_vars_init[~pd.isna(error_vars)] = error_vars[~pd.isna(error_vars)]
        error_covs_init = np.zeros((n_variables, n_variables))
        np.fill_diagonal(error_covs_init, error_vars_init)

        # calculate objective
        sigma_theta = loadings_init.dot(factor_cov_init_final) \
                                   .dot(loadings_init.T) + error_covs_init
        error = (np.log(np.linalg.det(sigma_theta)) +
                 np.trace(cov.dot(np.linalg.inv(sigma_theta)) -
                 np.log(np.linalg.det(cov)) - n_variables))

        return error

    def analyze(self,
                data,
                model,
                is_cov=False,
                fix_first=True,
                bounds=None,
                maxiter=50,
                tol=None):
        """
        Perform confirmatory factor analysis

        Parameters
        ----------
        data : pd.DataFrame or np.array
            The data to use for confirmatory
            factor analysis. If this is just a
            covariance matrix, make sure `is_cov`
            is set to True.
        model : dict
            A dictionary specifying the model
            inputs. In general, the `model`
            can have the following keys:
                - `loadings` : dict or pd.DataFrame
                    A dictionary where the keys are factor names
                    and the values are lists of variables that load
                    on each of those factors. Alternatively, you can
                    specify a data frame, where the columns are factor
                    names, the index is variable names, and the values
                    are 0s and 1s that indicates whether a variable
                    loads on a particular factor.
                - factor_covs : list or None, optional
                    A list of factor covariances. If None, no
                    covariance constraints will be imposed on the
                    confirmatory factor analysis model. This list
                    should be in the following format:
                        ```
                        [cov(x1, x2'), cov(x1, x3), cov(x2, x3), ...]
                        ```
                - factor_vars : list or None, optional
                    A list of factor variances. If None, no
                    factor variance constraints will be imposed on the
                    confirmatory factor analysis model.
                - error_vars : list or None, optional
                    A list of error variances. If None, no
                    error variance constraints will be imposed on the
                    confirmatory factor analysis model.
        is_cov : bool, optional
            Whether the input `data` is a
            covariance matrix. If False,
            assume it is the full data set
            Defaults to False.
        fix_first : bool, optional
            Whether to fix the first variable loading
            on a given factor to 1.
            Defaults to True.
        bounds : list of tuples or None, optional
            A list of minimum and maximum
            boundaries for each element of the
            input array. This must equal `x0`,
            which is the input array from your
            parsed and combined `model`. If None,
            only the loading matrix will be bounded
            within the range (0, 1).
            Defaults to None.
        maxiter : int, optional
            The maximum number of iterations
            for the optimization routine.
            Defaults to 50.
        tol : float or None, optional
            The tolerance for convergence.
            Defaults to None.

        Returns
        -------
        results
        """
        if not is_cov:
            data = data.cov() * ((data.shape[0] - 1) / data.shape[0])

        (loadings,
         error_vars,
         factor_vars,
         factor_covs,
         variable_names,
         factor_names,
         n_factors,
         n_variables,
         n_lower_diag) = ModelParser().parse(model)

        loading_init = loadings.copy()
        error_vars_init = np.full((error_vars.shape[0], 1), 0.5)
        factor_vars_init = np.full((factor_vars.shape[0], 1), 0.5)
        factor_covs_init = np.full((factor_covs.shape[0], 1), 0.05)

        x0 = self.combine(loading_init,
                          error_vars_init,
                          factor_vars_init,
                          factor_covs_init,
                          n_factors,
                          n_variables,
                          n_lower_diag)

        if bounds is None:
            bounds = [(0, 1) for _ in range(loading_init.shape[0] *
                                            loading_init.shape[1])]
            bounds += [(None, None) for _ in range(error_vars_init.shape[0])]
            bounds += [(None, None) for _ in range(factor_vars_init.shape[0])]
            bounds += [(None, None) for _ in range(factor_covs_init.shape[0])]

        error_msg = ('The length of `bounds` must equal the length of your '
                     'input array `x0`: {} != {}.'.format(len(bounds), len(x0)))
        assert len(bounds) == len(x0), error_msg

        res = minimize(self.objective, x0,
                       method='L-BFGS-B',
                       options={'maxiter': maxiter, 'disp': True},
                       bounds=bounds,
                       args=(data,
                             loadings,
                             error_vars,
                             factor_vars,
                             factor_covs,
                             n_factors,
                             n_variables,
                             n_lower_diag,
                             fix_first))

        (loadings,
         error_vars,
         factor_vars,
         factor_covs) = self.split(res.x, n_factors, n_variables, n_lower_diag)

        factor_covs_full = fill_lower_diag(factor_covs)
        factor_covs_full += factor_covs_full.T
        np.fill_diagonal(factor_covs_full, factor_vars)

        self.loadings = pd.DataFrame(loadings, columns=factor_names, index=variable_names)
        self.error_vars = pd.DataFrame(error_vars, columns=['error_vars'], index=variable_names)
        self.factor_covs = pd.DataFrame(factor_covs_full, columns=factor_names, index=factor_names)
