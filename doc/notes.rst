.. _important_notes:

Important notes
===============

1. It is possible that `factor_analyzer` may return the loading for a factor
   that has all negative entries whereas SPSS/R may return the same loading
   with all positive entries. This is not a bug. This can happen if the
   eigenvalue decomposition returns an eigenvector with all negative entries,
   which is not unusual since if :math:`v` is an eigenvector, then so is
   :math:`\alpha * v`, where :math:`\alpha` is any scalar (:math:`\ne 0`).
   Additionally, signs on factor loadings are also kind of meaningless because
   all they do is flip the (already arbitrary) interpretation of the latent
   factor. For more details, please refer to
   `this Github issue <https://github.com/EducationalTestingService/factor_analyzer/issues/89>`__.
2. When using equamax rotation, you must compute the correct value of
   :math:`\kappa` yourself and pass it using the `rotation_kwargs` argument.
   This is different from SPSS which computes the value of :math:`\kappa`
   internally. For more details, please refer to
   `this Github issue <https://github.com/EducationalTestingService/factor_analyzer/issues/90>`__.
