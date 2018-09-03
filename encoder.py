from __future__ import print_function, division, absolute_import
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d
# from hmmlearn.preprocessing.label import _check_numpy_unicode_bug
import numpy as np
import pandas as pd


def _get_unseen():
    """Basically just a static method
    instead of a class attribute to avoid
    someone accidentally changing it."""
    return 99999

class SafeLabelEncoder(LabelEncoder):
    """An extension of LabelEncoder that will
    not throw an exception for unseen data, but will
    instead return a default value of 99999
    Attributes
    ----------
    classes_ : the classes that are encoded
    """

    def transform(self, y):
        """Perform encoding if already fit.
        Parameters
        ----------
        y : array_like, shape=(n_samples,)
            The array to encode
        Returns
        -------
        e : array_like, shape=(n_samples,)
            The encoded array
        """
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        classes = np.unique(y)
        #         _check_numpy_unicode_bug(classes)

        # Check not too many:
        unseen = _get_unseen()
        if len(classes) >= unseen:
            raise ValueError('Too many factor levels in feature. Max is %i' % unseen)

        e = np.array([
            np.searchsorted(self.classes_, x) if x in self.classes_ else unseen
            for x in y
        ])

        return e