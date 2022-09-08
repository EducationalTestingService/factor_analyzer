"""
Class to perform various rotations of factor loading matrices.

:author: Jeremy Biggs (jeremy.m.biggs@gmail.com)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: Educational Testing Service
:date: 2022-09-05
"""

import numpy as np
import scipy as sp
from sklearn.base import BaseEstimator

ORTHOGONAL_ROTATIONS = ["varimax", "oblimax", "quartimax", "equamax", "geomin_ort"]

OBLIQUE_ROTATIONS = ["promax", "oblimin", "quartimin", "geomin_obl"]

POSSIBLE_ROTATIONS = ORTHOGONAL_ROTATIONS + OBLIQUE_ROTATIONS


class Rotator(BaseEstimator):
    """
    Perform rotations on an unrotated factor loading matrix.

    The Rotator class takes an (unrotated) factor loading matrix and
    performs one of several rotations.

    Parameters
    ----------
    method : str, optional
        The factor rotation method. Options include:
            (a) varimax (orthogonal rotation)
            (b) promax (oblique rotation)
            (c) oblimin (oblique rotation)
            (d) oblimax (orthogonal rotation)
            (e) quartimin (oblique rotation)
            (f) quartimax (orthogonal rotation)
            (g) equamax (orthogonal rotation)
            (h) geomin_obl (oblique rotation)
            (i) geomin_ort (orthogonal rotation)

        Defaults to 'varimax'.
    normalize : bool or None, optional
        Whether to perform Kaiser normalization and de-normalization prior
        to and following rotation. Used for 'varimax' and 'promax' rotations.
        If ``None``, default for 'promax' is ``False``, and default for
        'varimax' is ``True``.
        Defaults to ``None``.
    power : int, optional
        The exponent to which to raise the promax loadings (minus 1).
        Numbers should generally range from 2 to 4.
        Defaults to 4.
    kappa : float, optional
        The kappa value for the 'equamax' objective. Ignored if the method
        is not 'equamax'.
        Defaults to 0.
    gamma : int, optional
        The gamma level for the 'oblimin' objective. Ignored if the method
        is not 'oblimin'.
        Defaults to 0.
    delta : float, optional
        The delta level for 'geomin' objectives. Ignored if the method is
        not 'geomin_*'.
        Defaults to 0.01.
    max_iter : int, optional
        The maximum number of iterations. Used for 'varimax' and 'oblique'
        rotations.
        Defaults to 1000.
    tol : float, optional
        The convergence threshold. Used for 'varimax' and 'oblique' rotations.
        Defaults to 1e-5.

    Attributes
    ----------
    loadings_ : :obj:`numpy.ndarray`, shape (``n_features``, ``n_factors``)
        The loadings matrix.
    rotation_ : :obj:`numpy.ndarray`, shape (``n_factors``, ``n_factors``)
        The rotation matrix.
    phi_ : :obj:`numpy.ndarray` or None
        The factor correlations matrix. This only exists if ``method`` is
        'oblique'.

    Notes
    -----
    Most of the rotations in this class are ported from R's ``GPARotation``
    package.

    References
    ----------
    [1] https://cran.r-project.org/web/packages/GPArotation/index.html

    Examples
    --------
    >>> import pandas as pd
    >>> from factor_analyzer import FactorAnalyzer, Rotator
    >>> df_features = pd.read_csv('test02.csv')
    >>> fa = FactorAnalyzer(rotation=None)
    >>> fa.fit(df_features)
    >>> rotator = Rotator()
    >>> rotator.fit_transform(fa.loadings_)
    array([[-0.07693215,  0.04499572,  0.76211208],
           [ 0.01842035,  0.05757874,  0.01297908],
           [ 0.06067925,  0.70692662, -0.03311798],
           [ 0.11314343,  0.84525117, -0.03407129],
           [ 0.15307233,  0.5553474 , -0.00121802],
           [ 0.77450832,  0.1474666 ,  0.20118338],
           [ 0.7063001 ,  0.17229555, -0.30093981],
           [ 0.83990851,  0.15058874, -0.06182469],
           [ 0.76620579,  0.1045194 , -0.22649615],
           [ 0.81372945,  0.20915845,  0.07479506]])
    """

    def __init__(
        self,
        method="varimax",
        normalize=True,
        power=4,
        kappa=0,
        gamma=0,
        delta=0.01,
        max_iter=500,
        tol=1e-5,
    ):
        """Initialize the rotator class."""
        self.method = method
        self.normalize = normalize
        self.power = power
        self.kappa = kappa
        self.gamma = gamma
        self.delta = delta
        self.max_iter = max_iter
        self.tol = tol

        self.loadings_ = None
        self.rotation_ = None
        self.phi_ = None

    def _oblimax_obj(self, loadings):  # noqa: D401
        """
        The Oblimax function objective.

        Parameters
        ----------
        loadings : array-like
            The loading matrix

        Returns
        -------
        gradient_dict : dict
            A dictionary containing the following keys:
            (1) ``grad`` : :obj:`numpy.ndarray`, containing the gradients.
            (2) ``criterion`` : float, containing the criterion for the objective.
        """
        gradient = -(
            4 * loadings**3 / (np.sum(loadings**4))
            - 4 * loadings / (np.sum(loadings**2))
        )
        criterion = np.log(np.sum(loadings**4)) - 2 * np.log(np.sum(loadings**2))
        return {"grad": gradient, "criterion": criterion}

    def _quartimax_obj(self, loadings):
        """
        Quartimax function objective.

        Parameters
        ----------
        loadings : array-like
            The loading matrix.

        Returns
        -------
        gradient_dict : dict
            A dictionary containing the following keys:
            (1) ``grad`` : :obj:`numpy.ndarray`, containing the gradients.
            (2) ``criterion`` : float, containing the criterion for the objective.
        """
        gradient = -(loadings**3)
        criterion = -np.sum(np.diag(np.dot((loadings**2).T, loadings**2))) / 4
        return {"grad": gradient, "criterion": criterion}

    def _oblimin_obj(self, loadings):  # noqa: D401
        """
        The Oblimin function objective.

        Parameters
        ----------
        loadings : array-like
            The loading matrix

        Returns
        -------
        gradient_dict : dict
            A dictionary containing the following keys:
            (1) ``grad`` : :obj:`numpy.ndarray`, containing the gradients.
            (2) ``criterion`` : float, containing the criterion for the objective.
        """
        X = np.dot(loadings**2, np.eye(loadings.shape[1]) != 1)
        if self.gamma != 0:
            p = loadings.shape[0]
            X = np.diag(np.full(1, p)) - np.dot(np.zeros((p, p)), X)
        gradient = loadings * X
        criterion = np.sum(loadings**2 * X) / 4
        return {"grad": gradient, "criterion": criterion}

    def _quartimin_obj(self, loadings):  # noqa: D401
        """
        The Quartimin function objective.

        Parameters
        ----------
        loadings : array-like
            The loading matrix.

        Returns
        -------
        gradient_dict : dict
            A dictionary containing the following keys:
            (1) ``grad`` : :obj:`numpy.ndarray`, containing the gradients.
            (2) ``criterion`` : float, containing the criterion for the objective.
        """
        X = np.dot(loadings**2, np.eye(loadings.shape[1]) != 1)
        gradient = loadings * X
        criterion = np.sum(loadings**2 * X) / 4
        return {"grad": gradient, "criterion": criterion}

    def _equamax_obj(self, loadings):  # noqa: D401
        """
        The Equamax function objective.

        Parameters
        ----------
        loadings : array-like
            The loading matrix.

        Returns
        -------
        gradient_dict : dict
            A dictionary containing the following keys:
            (1) ``grad`` : :obj:`numpy.ndarray`, containing the gradients.
            (2) ``criterion`` : float, containing the criterion for the objective.
        """
        p, k = loadings.shape

        N = np.ones(k) - np.eye(k)
        M = np.ones(p) - np.eye(p)

        loadings_squared = loadings**2
        f1 = (
            (1 - self.kappa)
            * np.sum(np.diag(np.dot(loadings_squared.T, np.dot(loadings_squared, N))))
            / 4
        )
        f2 = (
            self.kappa
            * np.sum(np.diag(np.dot(loadings_squared.T, np.dot(M, loadings_squared))))
            / 4
        )

        gradient = (1 - self.kappa) * loadings * np.dot(
            loadings_squared, N
        ) + self.kappa * loadings * np.dot(M, loadings_squared)

        criterion = f1 + f2
        return {"grad": gradient, "criterion": criterion}

    def _geomin_obj(self, loadings):  # noqa: D401
        """
        The Geomin function objective.

        Parameters
        ----------
        loadings : array-like
            The loading matrix.

        Returns
        -------
        gradient_dict : dict
            A dictionary containing the following keys:
            (1) ``grad`` : :obj:`numpy.ndarray`, containing the gradients.
            (2) ``criterion`` : float, containing the criterion for the objective.
        """
        p, k = loadings.shape

        loadings2 = loadings**2 + self.delta

        pro = np.exp(np.log(loadings2).sum(1) / k)
        rep = np.repeat(pro, k, axis=0).reshape(p, k)

        gradient = (2 / k) * (loadings / loadings2) * rep
        criterion = np.sum(pro)
        return {"grad": gradient, "criterion": criterion}

    def _oblique(self, loadings, method):
        """
        Perform oblique rotations, except 'promax'.

        A generic function for performing all oblique rotations, except for
        promax, which is implemented separately.

        Parameters
        ----------
        loadings : array-like
            The loading matrix
        method : str
            The obligue rotation method to use.

        Returns
        -------
        loadings : :obj:`numpy.ndarray`, shape (``n_features``, ``n_factors``)
            The loadings matrix
        rotation_mtx : :obj:`numpy.ndarray`, shape (``n_factors``, ``n_factors``)
            The rotation matrix
        phi : :obj:`numpy.ndarray`, shape (``n_factors``, ``n_factors``)
            The factor correlations matrix. This only exists if the
            rotation is oblique.
        """
        if method == "oblimin":
            objective = self._oblimin_obj
        elif method == "quartimin":
            objective = self._quartimin_obj
        elif method == "geomin_obl":
            objective = self._geomin_obj

        # initialize the rotation matrix
        _, n_cols = loadings.shape
        rotation_matrix = np.eye(n_cols)

        # default alpha level
        alpha = 1
        rotation_matrix_inv = np.linalg.inv(rotation_matrix)
        new_loadings = np.dot(loadings, rotation_matrix_inv.T)

        obj = objective(new_loadings)
        gradient = -np.dot(new_loadings.T, np.dot(obj["grad"], rotation_matrix_inv)).T
        criterion = obj["criterion"]

        obj_t = objective(new_loadings)

        # main iteration loop, up to `max_iter`, calculate the gradient
        for _ in range(0, self.max_iter + 1):
            gradient_new = gradient - np.dot(
                rotation_matrix,
                np.diag(np.dot(np.ones(gradient.shape[0]), rotation_matrix * gradient)),
            )
            s = np.sqrt(np.sum(np.diag(np.dot(gradient_new.T, gradient_new))))

            if s < self.tol:
                break

            alpha = 2 * alpha

            # calculate the Hessian of the objective function
            for _ in range(0, 11):
                X = rotation_matrix - alpha * gradient_new

                v = 1 / np.sqrt(np.dot(np.ones(X.shape[0]), X**2))
                new_rotation_matrix = np.dot(X, np.diag(v))
                new_loadings = np.dot(loadings, np.linalg.inv(new_rotation_matrix).T)

                obj_t = objective(new_loadings)
                improvement = criterion - obj_t["criterion"]

                if improvement > 0.5 * s**2 * alpha:
                    break

                alpha = alpha / 2

            rotation_matrix = new_rotation_matrix
            criterion = obj_t["criterion"]
            gradient = -np.dot(
                np.dot(new_loadings.T, obj_t["grad"]),
                np.linalg.inv(new_rotation_matrix),
            ).T

        # calculate phi
        phi = np.dot(rotation_matrix.T, rotation_matrix)

        # convert loadings matrix to data frame
        loadings = new_loadings.copy()
        return loadings, rotation_matrix, phi

    def _orthogonal(self, loadings, method):
        """
        Perform orthogonal rotations, except 'varimax'.

        A generic function for performing all orthogonal rotations, except for
        varimax, which is implemented separately.

        Parameters
        ----------
        loadings : :obj:`numpy.ndarray`
            The loading matrix
        method : str
            The orthogonal rotation method to use.

        Returns
        -------
        loadings : :obj:`numpy.ndarray`
            The loadings matrix
        rotation_mtx : :obj:`numpy.ndarray`, shape (``n_factors``, ``n_factors``)
            The rotation matrix
        """
        if method == "oblimax":
            objective = self._oblimax_obj
        elif method == "quartimax":
            objective = self._quartimax_obj
        elif method == "equamax":
            objective = self._equamax_obj
        elif method == "geomin_ort":
            objective = self._geomin_obj

        arr = loadings.copy()

        # initialize the rotation matrix
        _, n_cols = arr.shape
        rotation_matrix = np.eye(n_cols)

        # default alpha level
        alpha = 1
        new_loadings = np.dot(arr, rotation_matrix)

        obj = objective(new_loadings)
        gradient = np.dot(arr.T, obj["grad"])
        criterion = obj["criterion"]

        obj_t = objective(new_loadings)

        # main iteration loop, up to `max_iter`, calculate the gradient
        for _ in range(0, self.max_iter + 1):
            M = np.dot(rotation_matrix.T, gradient)
            S = (M + M.T) / 2
            gradient_new = gradient - np.dot(rotation_matrix, S)
            s = np.sqrt(np.sum(np.diag(np.dot(gradient_new.T, gradient_new))))

            if s < self.tol:
                break

            alpha = 2 * alpha

            # calculate the Hessian of the objective function
            for _ in range(0, 11):
                X = rotation_matrix - alpha * gradient_new
                U, _, V = np.linalg.svd(X)
                new_rotation_matrix = np.dot(U, V)
                new_loadings = np.dot(arr, new_rotation_matrix)

                obj_t = objective(new_loadings)

                if obj_t["criterion"] < (criterion - 0.5 * s**2 * alpha):
                    break

                alpha = alpha / 2

            rotation_matrix = new_rotation_matrix
            criterion = obj_t["criterion"]
            gradient = np.dot(arr.T, obj_t["grad"])

        # convert loadings matrix to data frame
        loadings = new_loadings.copy()
        return loadings, rotation_matrix

    def _varimax(self, loadings):
        """
        Perform varimax (orthogonal) rotation, with optional Kaiser normalization.

        Parameters
        ----------
        loadings : array-like
            The loading matrix.

        Returns
        -------
        loadings : :obj:`numpy.ndarray`, shape (``n_features``, ``n_factors``)
            The loadings matrix.
        rotation_mtx : :obj:`numpy.ndarray`, shape (``n_factors``, ``n_factors``)
            The rotation matrix.
        """
        X = loadings.copy()
        n_rows, n_cols = X.shape
        if n_cols < 2:
            return X

        # normalize the loadings matrix
        # using sqrt of the sum of squares (Kaiser)
        if self.normalize:
            normalized_mtx = np.apply_along_axis(
                lambda x: np.sqrt(np.sum(x**2)), 1, X.copy()
            )
            X = (X.T / normalized_mtx).T

        # initialize the rotation matrix
        # to N x N identity matrix
        rotation_mtx = np.eye(n_cols)

        d = 0
        for _ in range(self.max_iter):

            old_d = d

            # take inner product of loading matrix
            # and rotation matrix
            basis = np.dot(X, rotation_mtx)

            # transform data for singular value decomposition using updated formula :
            # B <- t(x) %*% (z^3 - z %*% diag(drop(rep(1, p) %*% z^2))/p)
            diagonal = np.diag(np.squeeze(np.repeat(1, n_rows).dot(basis**2)))
            transformed = X.T.dot(basis**3 - basis.dot(diagonal) / n_rows)

            # perform SVD on
            # the transformed matrix
            U, S, V = np.linalg.svd(transformed)

            # take inner product of U and V, and sum of S
            rotation_mtx = np.dot(U, V)
            d = np.sum(S)

            # check convergence
            if d < old_d * (1 + self.tol):
                break

        # take inner product of loading matrix
        # and rotation matrix
        X = np.dot(X, rotation_mtx)

        # de-normalize the data
        if self.normalize:
            X = X.T * normalized_mtx
        else:
            X = X.T

        # convert loadings matrix to data frame
        loadings = X.T.copy()
        return loadings, rotation_mtx

    def _promax(self, loadings):
        """
        Perform promax (oblique) rotation, with optional Kaiser normalization.

        Parameters
        ----------
        loadings : array-like
            The loading matrix

        Returns
        -------
        loadings : :obj:`numpy.ndarray`, shape (``n_features``, ``n_factors``)
            The loadings matrix
        rotation_mtx : :obj:`numpy.ndarray`, shape (``n_factors``, ``n_factors``)
            The rotation matrix
        phi : :obj:`numpy.ndarray` or None, shape (``n_factors``, ``n_factors``)
            The factor correlations matrix. This only exists if the rotation
            is oblique.
        """
        X = loadings.copy()
        _, n_cols = X.shape
        if n_cols < 2:
            return X

        if self.normalize:
            # pre-normalization is done in R's
            # `kaiser()` function when rotate='Promax'.
            array = X.copy()
            h2 = sp.diag(np.dot(array, array.T))
            h2 = np.reshape(h2, (h2.shape[0], 1))
            weights = array / sp.sqrt(h2)

        else:
            weights = X.copy()

        # first get varimax rotation
        X, rotation_mtx = self._varimax(weights)
        Y = X * np.abs(X) ** (self.power - 1)

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

        if self.normalize:
            # post-normalization is done in R's
            # `kaiser()` function when rotate='Promax'
            z = z * sp.sqrt(h2)

        rotation_mtx = sp.dot(rotation_mtx, coef)

        coef_inv = np.linalg.inv(coef)
        phi = np.dot(coef_inv, coef_inv.T)

        # convert loadings matrix to data frame
        loadings = z.copy()
        return loadings, rotation_mtx, phi

    def fit(self, X, y=None):
        """
        Compute the factor rotation.

        Parameters
        ----------
        X : array-like
            The factor loading matrix, shape (``n_features``, ``n_factors``)
        y : ignored

        Returns
        -------
        self

        Example
        -------
        >>> import pandas as pd
        >>> from factor_analyzer import FactorAnalyzer, Rotator
        >>> df_features = pd.read_csv('test02.csv')
        >>> fa = FactorAnalyzer(rotation=None)
        >>> fa.fit(df_features)
        >>> rotator = Rotator()
        >>> rotator.fit(fa.loadings_)
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        """
        Compute the factor rotation, and return the new loading matrix.

        Parameters
        ----------
        X : array-like
            The factor loading matrix, shape (``n_features``, ``n_factors``)
        y : Ignored

        Returns
        -------
        loadings_ : :obj:`numpy,ndarray`, shape (``n_features``, ``n_factors``)
            The loadings matrix.

        Raises
        ------
        ValueError
            If ``method`` is not in the list of acceptable methods.

        Example
        -------
        >>> import pandas as pd
        >>> from factor_analyzer import FactorAnalyzer, Rotator
        >>> df_features = pd.read_csv('test02.csv')
        >>> fa = FactorAnalyzer(rotation=None)
        >>> fa.fit(df_features)
        >>> rotator = Rotator()
        >>> rotator.fit_transform(fa.loadings_)
        array([[-0.07693215,  0.04499572,  0.76211208],
               [ 0.01842035,  0.05757874,  0.01297908],
               [ 0.06067925,  0.70692662, -0.03311798],
               [ 0.11314343,  0.84525117, -0.03407129],
               [ 0.15307233,  0.5553474 , -0.00121802],
               [ 0.77450832,  0.1474666 ,  0.20118338],
               [ 0.7063001 ,  0.17229555, -0.30093981],
               [ 0.83990851,  0.15058874, -0.06182469],
               [ 0.76620579,  0.1045194 , -0.22649615],
               [ 0.81372945,  0.20915845,  0.07479506]])
        """
        # default phi to None
        # it will only be calculated
        # for oblique rotations
        phi = None
        method = self.method.lower()
        if method == "varimax":
            (new_loadings, new_rotation_mtx) = self._varimax(X)

        elif method == "promax":
            (new_loadings, new_rotation_mtx, phi) = self._promax(X)

        elif method in OBLIQUE_ROTATIONS:
            (new_loadings, new_rotation_mtx, phi) = self._oblique(X, method)

        elif method in ORTHOGONAL_ROTATIONS:
            (new_loadings, new_rotation_mtx) = self._orthogonal(X, method)

        else:
            raise ValueError(
                "The value for `method` must be one of the "
                "following: {}.".format(", ".join(POSSIBLE_ROTATIONS))
            )

        (self.loadings_, self.rotation_, self.phi_) = (
            new_loadings,
            new_rotation_mtx,
            phi,
        )
        return self.loadings_
