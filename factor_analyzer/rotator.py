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


POSSIBLE_ROTATIONS = ['varimax', 'promax',
                      'oblimax', 'quartimax']


class Rotator:
    """
    The Rotator class.
    """

    @staticmethod
    def _oblimax_obj(loadings):
        """
        The Oblimax function

        Parameters
        ----------
        loadings : array-like
            The loading matrix

        Returns
        -------
        dict
        """
        gradient = -(4 * loadings**3 / (np.sum(loadings**4)) - 4 * loadings /
                     (np.sum(loadings**2)))
        error = (np.log(np.sum(loadings**4)) - 2 * np.log(np.sum(loadings**2)))
        return {'grad': gradient, 'error': error}

    @staticmethod
    def _quartimax_obj(loadings):
        """
        Quartimax function

        Parameters
        ----------
        loadings : array-like
            The loading matrix

        Returns
        -------
        dict
        """
        gradient = -loadings**3
        error = -np.sum(np.diag(np.dot((loadings**2).T, loadings**2))) / 4
        return {'grad': gradient, 'error': error}

    def rotate(self, loadings, method='varimax', **kwargs):
        """
        Rotate the factor loading matrix.

        Parameters
        ----------
        loadings : pd.DataFrame
            The loadings matrix from your factor analysis.
        method : str
            The factor rotation method. Options include:

                (a) varimax (orthogonal rotation)
                (b) promax (oblique rotation)

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
        method = method.lower()
        if method == 'varimax':
            (new_loadings,
             new_rotation_mtx) = self.varimax(loadings, **kwargs)

        elif method == 'promax':
            (new_loadings,
             new_rotation_mtx) = self.promax(loadings, **kwargs)

        elif method == 'oblimax':
            (new_loadings,
             new_rotation_mtx) = self.orthogonal(loadings, self._oblimax_obj, **kwargs)

        elif method == 'quartimax':
            (new_loadings,
             new_rotation_mtx) = self.orthogonal(loadings, self._quartimax_obj, **kwargs)

        else:
            raise ValueError("The value for `method` must be one of the "
                             "following: {}.".format(', '.join(POSSIBLE_ROTATIONS)))

        return new_loadings, new_rotation_mtx

    def oblique(self,
                loadings,
                objective,
                max_iter=1000,
                tolerance=1e-5,
                **kwargs):
        """
        A generic function for performing
        oblique rotations, except for
        promax, which is implemented
        differently.

        Parameters
        ----------
        loadings : pd.DataFrame
            The original loadings matrix
        objective : function
            The function for a given orthogonal
            rotation method.
        max_iter : int, optional
            The maximum number of iterations.
            Defaults to `1000`.
        tolerance : float, optional
            The convergence threshold.
            Defaults to `1e-5`.

        Return
        ------
        loadings : np.array
            The loadings matrix
            (n_cols, n_factors)
        rotation_mtx : np.array
            The rotation matrix
            (n_factors, n_factors)
        """
        df = loadings.copy()

        column_names = df.columns.values
        index_names = df.index.values

        n_rows, n_cols = loadings.shape
        rotation_matrix = np.eye(n_cols)

        al = 1
        rotation_matrix_inv = np.linalg.inv(rotation_matrix)
        new_loadings = np.dot(loadings, rotation_matrix_inv.T)

        obj = objective(new_loadings)
        gradient = -np.dot(new_loadings.T, np.dot(obj['grad'], rotation_matrix_inv)).T
        error = obj['error']

        obj_t = objective(new_loadings)

        for i in range(0, max_iter + 1):
            gradient_new = gradient - np.dot(rotation_matrix,
                                             np.diag(np.dot(np.ones(gradient.shape[0]),
                                                            rotation_matrix * gradient)))
            s = np.sqrt(np.sum(np.diag(np.dot(gradient_new.T, gradient_new))))

            if (s < tolerance):
                break

            al = 2 * al

            for j in range(0, 11):
                X = rotation_matrix - al * gradient_new

                v = 1 / np.sqrt(np.dot(np.ones(X.shape[0]), X**2))
                new_rotation_matrix = np.dot(X, np.diag(v))
                new_loadings = np.dot(loadings, np.linalg.inv(new_rotation_matrix).T)

                obj_t = objective(new_loadings)
                improvement = error - obj_t['error']

                if (improvement > 0.5 * s**2 * al):
                    break

                al = al / 2

            rotation_matrix = new_rotation_matrix
            error = obj_t['error']
            gradient = -np.dot(np.dot(new_loadings.T, obj_t['grad']),
                               np.linalg.inv(new_rotation_matrix)).T

        # convert loadings matrix to data frame
        loadings = pd.DataFrame(new_loadings,
                                columns=column_names,
                                index=index_names)

        return loadings, rotation_matrix

    def orthogonal(self,
                   loadings,
                   objective,
                   max_iter=1000,
                   tolerance=1e-5,
                   **kwargs):
        """
        A generic function for performing
        orthogonal rotations, except for
        varimax, which is implemented
        differently.

        Parameters
        ----------
        loadings : pd.DataFrame
            The original loadings matrix
        objective : function
            The function for a given orthogonal
            rotation method.
        max_iter : int, optional
            The maximum number of iterations.
            Defaults to `1000`.
        tolerance : float, optional
            The convergence threshold.
            Defaults to `1e-5`.

        Return
        ------
        loadings : np.array
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

        al = 1
        new_loadings = np.dot(df, rotation_matrix)

        obj = objective(new_loadings)
        gradient = np.dot(df.T, obj['grad'])
        error = obj['error']

        obj_t = objective(new_loadings)

        for i in range(0, max_iter + 1):
            M = np.dot(rotation_matrix.T, gradient)
            S = (M + M.T) / 2
            gradient_new = gradient - np.dot(rotation_matrix, S)
            s = np.sqrt(np.sum(np.diag(np.dot(gradient_new.T, gradient_new))))

            if (s < tolerance):
                break

            al = 2 * al

            for j in range(0, 11):
                X = rotation_matrix - al * gradient_new
                U, D, V = np.linalg.svd(X)
                new_rotation_matrix = np.dot(U, V)
                new_loadings = np.dot(df, new_rotation_matrix)

                obj_t = objective(new_loadings)

                if (obj_t['error'] < (error - 0.5 * s**2 * al)):
                    break

                al = al / 2

            rotation_matrix = new_rotation_matrix
            error = obj_t['error']
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
        normalize : bool
            Whether to perform Kaiser normalization
            and de-normalization prior to and following
            rotation.
            Defaults to True.
        max_iter : int
            Maximum number of iterations.
            Defaults to 500.
        tolerance : float
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

        column_names = df.index.values
        index_names = df.columns.values

        n_rows, n_cols = df.shape

        if n_cols < 2:
            return df

        X = df.as_matrix()

        # normalize the loadings matrix
        # using sqrt of the sum of squares (Kaiser)
        if normalize:
            normalized_mtx = df.apply(lambda x: np.sqrt(sum(x**2)),
                                      axis=1).as_matrix()

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

        # convert loadings matrix to dataframe
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
        normalize : bool
            Whether to perform Kaiser normalization
            and de-normalization prior to and following
            rotation.
            Defaults to False.
        power : int
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

        column_names = df.index.values
        index_names = df.columns.values

        n_rows, n_cols = df.shape

        if n_cols < 2:
            return df

        if normalize:

            # pre-normalization is done in R's
            # `kaiser()` function when rotate='Promax'.
            array = df.as_matrix()
            h2 = sp.diag(np.dot(array, array.T))
            h2 = np.reshape(h2, (h2.shape[0], 1))
            weights = array / sp.sqrt(h2)

            # convert back to DataFrame for `varimax`
            weights = pd.DataFrame(weights,
                                   columns=index_names,
                                   index=column_names)
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

        # convert loadings matrix to DataFrame
        loadings = pd.DataFrame(z,
                                columns=index_names,
                                index=column_names)

        return loadings, rotation_mtx
