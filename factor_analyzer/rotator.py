"""
Rotator class to perform various
rotations of factor loading matrices.

:author: Jeremy Biggs (jbiggs@ets.org)

:date: 05/21/2018
:organization: ETS
"""

import numpy as np
import scipy as sp
import pandas as pd

ORTHOGONAL_ROTATIONS = ['varimax', 'oblimax', 'quartimax', 'equamax']

OBLIQUE_ROTATIONS = ['promax', 'oblimin', 'quartimin']

POSSIBLE_ROTATIONS = ORTHOGONAL_ROTATIONS + OBLIQUE_ROTATIONS


class Rotator:
    """
    The Rotator class takes an (unrotated)
    factor loading matrix and performs one
    of several rotations.

    Notes
    -----
    Most of the rotations in this class
    are ported from R's `GPARotation` package.

    References
    ----------
    [1] https://cran.r-project.org/web/packages/GPArotation/index.html

    Examples
    --------
    >>> import pandas as pd
    >>> from factor_analyzer import Rotator
    >>> unrotated_loadings = pd.read_csv('loading_uls_none_3_test01.csv')
    >>> rotator = Rotator()
    >>> loadings, rotate_mtx = rotator.rotate(unrotated_loadings, 'varimax')
    >>> loadings
               Factor1   Factor2   Factor3
    sex      -0.076925  0.044992  0.762026
    zygosity  0.018420  0.057579  0.012978
    moed      0.060674  0.706943 -0.033120
    faed      0.113147  0.845224 -0.034069
    faminc    0.153070  0.555351 -0.001220
    english   0.774515  0.147466  0.201190
    math      0.706296  0.172295 -0.300973
    socsci    0.839906  0.150589 -0.061835
    natsci    0.766202  0.104519 -0.226524
    vocab     0.813730  0.209159  0.074794
    """

    @staticmethod
    def _oblimax_obj(loadings, **kwargs):
        """
        The Oblimax function objective.

        Parameters
        ----------
        loadings : array-like
            The loading matrix

        Returns
        -------
        gradient_dict : dict
            A dictionary with
                - grad : np.array
                    The gradient.
                - criterion : float
                    The value of the criterion for the objective.
        """
        gradient = -(4 * loadings**3 / (np.sum(loadings**4)) - 4 * loadings /
                     (np.sum(loadings**2)))
        criterion = (np.log(np.sum(loadings**4)) - 2 * np.log(np.sum(loadings**2)))
        return {'grad': gradient, 'criterion': criterion}

    @staticmethod
    def _quartimax_obj(loadings, **kwargs):
        """
        Quartimax function objective.

        Parameters
        ----------
        loadings : array-like
            The loading matrix

        Returns
        -------
        gradient_dict : dict
            A dictionary with
                - grad : np.array
                    The gradient.
                - criterion : float
                    The value of the criterion for the objective.
        """
        gradient = -loadings**3
        criterion = -np.sum(np.diag(np.dot((loadings**2).T, loadings**2))) / 4
        return {'grad': gradient, 'criterion': criterion}

    @staticmethod
    def _oblimin_obj(loadings, gamma=0, **kwargs):
        """
        The Oblimin function objective.

        Parameters
        ----------
        loadings : array-like
            The loading matrix
        gamma : int, optional
            The gamma level for the objective.
            Defaults to 0.

        Returns
        -------
        gradient_dict : dict
            A dictionary with
                - grad : np.array
                    The gradient.
                - criterion : float
                    The value of the criterion for the objective.
        """
        X = np.dot(loadings**2, np.eye(loadings.shape[1]) != 1)
        if (0 != gamma):
            p = loadings.shape[0]
            X = np.diag(1, p) - np.dot(np.zeros((p, p)), X)
        gradient = loadings * X
        criterion = np.sum(loadings**2 * X) / 4
        return {'grad': gradient, 'criterion': criterion}

    @staticmethod
    def _quartimin_obj(loadings, **kwargs):
        """
        Quartimin function objective.

        Parameters
        ----------
        loadings : array-like
            The loading matrix

        Returns
        -------
        gradient_dict : dict
            A dictionary with
                - grad : np.array
                    The gradient.
                - criterion : float
                    The value of the criterion for the objective.
        """
        X = np.dot(loadings**2, np.eye(loadings.shape[1]) != 1)
        gradient = loadings * X
        criterion = np.sum(loadings**2 * X) / 4
        return {'grad': gradient, 'criterion': criterion}

    @staticmethod
    def _equamax_obj(loadings, kappa=0, **kwargs):
        """
        Equamax function objective.

        Parameters
        ----------
        loadings : array-like
            The loading matrix
        kappa : int, optional
            The kappa value for the objective
            Defaults to 0.

        Returns
        -------
        gradient_dict : dict
            A dictionary with
                - grad : np.array
                    The gradient.
                - criterion : float
                    The value of the criterion for the objective.
        """
        p, k = loadings.shape

        N = np.ones(k) - np.eye(k)
        M = np.ones(p) - np.eye(p)

        loadings_squared = loadings**2
        f1 = (1 - kappa) * np.sum(np.diag(np.dot(loadings_squared.T,
                                                 np.dot(loadings_squared, N)))) / 4
        f2 = kappa * np.sum(np.diag(np.dot(loadings_squared.T,
                                           np.dot(M, loadings_squared)))) / 4

        gradient = ((1 - kappa) * loadings * np.dot(loadings_squared, N) +
                    kappa * loadings * np.dot(M, loadings_squared))

        criterion = f1 + f2
        return {'grad': gradient, 'criterion': criterion}

    def oblique(self,
                loadings,
                objective,
                max_iter=1000,
                tolerance=1e-5,
                **kwargs):
        """
        A generic function for performing
        all oblique rotations, except for
        promax, which is implemented
        separately.

        Parameters
        ----------
        loadings : pd.DataFrame
            The original loadings matrix
        objective : function
            The function for a given orthogonal
            rotation method. Must return a dictionary
            with `grad` (gradient) and `criterion`
            (value of the objective criterion).
        max_iter : int, optional
            The maximum number of iterations.
            Defaults to `1000`.
        tolerance : float, optional
            The convergence threshold.
            Defaults to `1e-5`.
        kwargs
            Additional key word arguments
            are passed to the `objective`
            function.

        Return
        ------
        loadings : pd.DataFrame
            The loadings matrix
            (n_cols, n_factors)
        rotation_mtx : np.array
            The rotation matrix
            (n_factors, n_factors)
        """
        df = loadings.copy()

        column_names = df.columns.values
        index_names = df.index.values

        # initialize the rotation matrix
        n_rows, n_cols = loadings.shape
        rotation_matrix = np.eye(n_cols)

        # default alpha level
        alpha = 1

        rotation_matrix_inv = np.linalg.inv(rotation_matrix)
        new_loadings = np.dot(loadings, rotation_matrix_inv.T)

        obj = objective(new_loadings, **kwargs)
        gradient = -np.dot(new_loadings.T, np.dot(obj['grad'], rotation_matrix_inv)).T
        criterion = obj['criterion']

        obj_t = objective(new_loadings, **kwargs)

        # main iteration loop, up to `max_iter`, calculate the gradient
        for i in range(0, max_iter + 1):
            gradient_new = gradient - np.dot(rotation_matrix,
                                             np.diag(np.dot(np.ones(gradient.shape[0]),
                                                            rotation_matrix * gradient)))
            s = np.sqrt(np.sum(np.diag(np.dot(gradient_new.T, gradient_new))))

            if (s < tolerance):
                break

            alpha = 2 * alpha

            # calculate the Hessian of the objective function
            for j in range(0, 11):
                X = rotation_matrix - alpha * gradient_new

                v = 1 / np.sqrt(np.dot(np.ones(X.shape[0]), X**2))
                new_rotation_matrix = np.dot(X, np.diag(v))
                new_loadings = np.dot(loadings, np.linalg.inv(new_rotation_matrix).T)

                obj_t = objective(new_loadings, **kwargs)
                improvement = criterion - obj_t['criterion']

                if (improvement > 0.5 * s**2 * alpha):
                    break

                alpha = alpha / 2

            rotation_matrix = new_rotation_matrix
            criterion = obj_t['criterion']
            gradient = -np.dot(np.dot(new_loadings.T, obj_t['grad']),
                               np.linalg.inv(new_rotation_matrix)).T

        # calculate phi
        phi = np.dot(rotation_matrix.T, rotation_matrix)

        # convert loadings matrix to data frame
        loadings = pd.DataFrame(new_loadings,
                                columns=column_names,
                                index=index_names)

        return loadings, rotation_matrix, phi

    def orthogonal(self,
                   loadings,
                   objective,
                   max_iter=1000,
                   tolerance=1e-5,
                   **kwargs):
        """
        A generic function for performing
        all orthogonal rotations, except for
        varimax, which is implemented
        separately.

        Parameters
        ----------
        loadings : pd.DataFrame
            The original loadings matrix
        objective : function
            The function for a given orthogonal
            rotation method. Must return a dictionary
            with `grad` (gradient) and `criterion`
            (value of the objective criterion).
        max_iter : int, optional
            The maximum number of iterations.
            Defaults to `1000`.
        tolerance : float, optional
            The convergence threshold.
            Defaults to `1e-5`.
        kwargs
            Additional key word arguments
            are passed to the `objective`
            function.

        Return
        ------
        loadings : pd.DataFrame
            The loadings matrix
            (n_cols, n_factors)
        rotation_mtx : np.array
            The rotation matrix
            (n_factors, n_factors)
        """
        df = loadings.copy()

        column_names = df.columns.values
        index_names = df.index.values

        # initialize the rotation matrix
        n_rows, n_cols = df.shape
        rotation_matrix = np.eye(n_cols)

        # default alpha level
        alpha = 1

        new_loadings = np.dot(df, rotation_matrix)

        obj = objective(new_loadings, **kwargs)
        gradient = np.dot(df.T, obj['grad'])
        criterion = obj['criterion']

        obj_t = objective(new_loadings, **kwargs)

        # main iteration loop, up to `max_iter`, calculate the gradient
        for i in range(0, max_iter + 1):
            M = np.dot(rotation_matrix.T, gradient)
            S = (M + M.T) / 2
            gradient_new = gradient - np.dot(rotation_matrix, S)
            s = np.sqrt(np.sum(np.diag(np.dot(gradient_new.T, gradient_new))))

            if (s < tolerance):
                break

            alpha = 2 * alpha

            # calculate the Hessian of the objective function
            for j in range(0, 11):
                X = rotation_matrix - alpha * gradient_new
                U, D, V = np.linalg.svd(X)
                new_rotation_matrix = np.dot(U, V)
                new_loadings = np.dot(df, new_rotation_matrix)

                obj_t = objective(new_loadings, **kwargs)

                if (obj_t['criterion'] < (criterion - 0.5 * s**2 * alpha)):
                    break

                alpha = alpha / 2

            rotation_matrix = new_rotation_matrix
            criterion = obj_t['criterion']
            gradient = np.dot(df.T, obj_t['grad'])

        # convert loadings matrix to data frame
        loadings = pd.DataFrame(new_loadings,
                                columns=column_names,
                                index=index_names)

        return loadings, rotation_matrix

    def varimax(self, loadings, normalize=True, max_iter=500, tolerance=1e-5):
        """
        Perform varimax (orthogonal) rotation, with optional
        Kaiser normalization.

        Parameters
        ----------
        loadings : pd.DataFrame
            The loadings matrix to rotate.
        normalize : bool, optional
            Whether to perform Kaiser normalization
            and de-normalization prior to and following
            rotation.
            Defaults to True.
        max_iter : int, optional
            Maximum number of iterations.
            Defaults to 500.
        tolerance : float, optional
            The tolerance for convergence.
            Defaults to 1e-5.

        Return
        ------
        loadings : pd.DataFrame
            The loadings matrix
            (n_cols, n_factors)
        rotation_mtx : np.array
            The rotation matrix
            (n_factors, n_factors)
        """
        df = loadings.copy()

        # since we're transposing the matrix
        # later, we want to reverse the column
        # names and index names from the original
        # factor loading matrix at this point
        column_names = df.index.values
        index_names = df.columns.values

        n_rows, n_cols = df.shape

        if n_cols < 2:
            return df

        X = df.values

        # normalize the loadings matrix
        # using sqrt of the sum of squares (Kaiser)
        if normalize:
            normalized_mtx = df.apply(lambda x: np.sqrt(sum(x**2)),
                                      axis=1).values

            X = (X.T / normalized_mtx).T

        # initialize the rotation matrix
        # to N x N identity matrix
        rotation_mtx = np.eye(n_cols)

        d = 0
        for _ in range(max_iter):

            old_d = d

            # take inner product of loading matrix
            # and rotation matrix
            basis = np.dot(X, rotation_mtx)

            # transform data for singular value decomposition
            transformed = np.dot(X.T, basis**3 - (1.0 / n_rows) *
                                 np.dot(basis, np.diag(np.diag(np.dot(basis.T, basis)))))

            # perform SVD on
            # the transformed matrix
            U, S, V = np.linalg.svd(transformed)

            # take inner product of U and V, and sum of S
            rotation_mtx = np.dot(U, V)
            d = np.sum(S)

            # check convergence
            if old_d != 0 and d / old_d < 1 + tolerance:
                break

        # take inner product of loading matrix
        # and rotation matrix
        X = np.dot(X, rotation_mtx)

        # de-normalize the data
        if normalize:
            X = X.T * normalized_mtx

        else:
            X = X.T

        # convert loadings matrix to data frame
        loadings = pd.DataFrame(X,
                                columns=column_names,
                                index=index_names).T

        return loadings, rotation_mtx

    def promax(self, loadings, normalize=False, power=4):
        """
        Perform promax (oblique) rotation, with optional
        Kaiser normalization.

        Parameters
        ----------
        data : pd.DataFrame
            The loadings matrix to rotate.
        normalize : bool, optional
            Whether to perform Kaiser normalization
            and de-normalization prior to and following
            rotation.
            Defaults to False.
        power : int, optional
            The power to which to raise the varimax loadings
            (minus 1). Numbers should generally range form 2 to 4.
            Defaults to 4.

        Return
        ------
        loadings : pd.DataFrame
            The loadings matrix
            (n_cols, n_factors)
        rotation_mtx : np.array
            The rotation matrix
            (n_factors, n_factors)
        """
        df = loadings.copy()

        column_names = df.columns.values
        index_names = df.index.values

        n_rows, n_cols = df.shape

        if n_cols < 2:
            return df

        if normalize:

            # pre-normalization is done in R's
            # `kaiser()` function when rotate='Promax'.
            array = df.values
            h2 = sp.diag(np.dot(array, array.T))
            h2 = np.reshape(h2, (h2.shape[0], 1))
            weights = array / sp.sqrt(h2)

            # convert back to DataFrame for `varimax`
            weights = pd.DataFrame(weights,
                                   columns=column_names,
                                   index=index_names)
        else:
            weights = df.copy()

        # first get varimax rotation
        X, rotation_mtx = self.varimax(weights, normalize=normalize)
        Y = X * np.abs(X)**(power - 1)

        # fit linear regression model
        coef = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

        # calculate diagonal of inverse square
        try:
            diag_inv = sp.diag(sp.linalg.inv(sp.dot(coef.T, coef)))
        except np.linalg.LinAlgError:
            diag_inv = sp.diag(sp.linalg.pinv(sp.dot(coef.T, coef)))

        # transform and calculate inner products
        coef = sp.dot(coef, sp.diag(sp.sqrt(diag_inv)))
        z = sp.dot(X, coef)

        if normalize:

            # post-normalization is done in R's
            # `kaiser()` function when rotate='Promax'
            z = z * sp.sqrt(h2)

        rotation_mtx = sp.dot(rotation_mtx, coef)

        coef_inv = np.linalg.inv(coef)
        phi = np.dot(coef_inv, coef_inv.T)

        # convert loadings matrix to data frame
        loadings = pd.DataFrame(z,
                                columns=column_names,
                                index=index_names)

        return loadings, rotation_mtx, phi

    def rotate(self, loadings, method='varimax', **kwargs):
        """
        Rotate the factor loading matrix.

        Parameters
        ----------
        loadings : pd.DataFrame
            The loadings matrix from your factor analysis.
        method : str, optional
            The factor rotation method. Options include:

                (a) varimax (orthogonal rotation)
                (b) promax (oblique rotation)
                (c) oblimin (oblique rotation)
                (d) oblimax (orthogonal rotation)
                (e) quartimin (oblique rotation)
                (f) quartimax (orthogonal rotation)
                (g) equamax (orthogonal rotation)

            Defaults to 'varimax'.
        kwargs
            Additional key word arguments
            are passed to the rotation method.

        Returns
        -------
        loadings : pd.DataFrame
            The loadings matrix
            (n_cols, n_factors)
        rotation_mtx : np.array
            The rotation matrix
            (n_factors, n_factors)

        Raises
        ------
        ValueError
            If the `method` is not in the list of
            acceptable methods.
        """

        # default phi to None
        # it will only be calculated
        # for oblique rotations
        phi = None

        method = method.lower()
        if method == 'varimax':
            (new_loadings,
             new_rotation_mtx) = self.varimax(loadings, **kwargs)

        elif method == 'promax':
            (new_loadings,
             new_rotation_mtx, phi) = self.promax(loadings, **kwargs)

        elif method == 'oblimax':
            (new_loadings,
             new_rotation_mtx) = self.orthogonal(loadings, self._oblimax_obj, **kwargs)

        elif method == 'quartimax':
            (new_loadings,
             new_rotation_mtx) = self.orthogonal(loadings, self._quartimax_obj, **kwargs)

        elif method == 'oblimin':
            (new_loadings,
             new_rotation_mtx, phi) = self.oblique(loadings, self._oblimin_obj, **kwargs)

        elif method == 'quartimin':
            (new_loadings,
             new_rotation_mtx, phi) = self.oblique(loadings, self._quartimin_obj, **kwargs)

        elif method == 'equamax':
            (new_loadings,
             new_rotation_mtx) = self.orthogonal(loadings, self._equamax_obj, **kwargs)

        else:
            raise ValueError("The value for `method` must be one of the "
                             "following: {}.".format(', '.join(POSSIBLE_ROTATIONS)))

        return new_loadings, new_rotation_mtx, phi
