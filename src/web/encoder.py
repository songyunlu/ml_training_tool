from __future__ import print_function, division, absolute_import
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder, LeaveOneOutEncoder
from sklearn.utils.random import check_random_state

def _get_unseen():
    """Basically just a static method
    instead of a class attribute to avoid
    someone accidentally changing it."""
    return -2
    # return 99999

class SafeLabelEncoder(LabelEncoder):
    """An extension of LabelEncoder that will
    not throw an exception for unseen data, but will
    instead return a default value
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
        # _check_numpy_unicode_bug(classes)

        # Check not too many:
        unseen = _get_unseen()
        # if len(classes) >= unseen:
        #     raise ValueError('Too many factor levels in feature. Max is %i' % unseen)

        e = np.array([
            np.searchsorted(self.classes_, x) if x in self.classes_ else unseen
            for x in y
        ])

        return e


class EnhancedTargetEncoder(TargetEncoder):

    def target_encode(self, X_in, y, mapping=None, cols=None, impute_missing=True, handle_unknown='impute', min_samples_leaf=1, smoothing_in=1.0):
        X = X_in.copy(deep=True)
        if cols is None:
            cols = X.columns.values

        if mapping is not None:
            mapping_out = mapping
            for switch in mapping:
                column = switch.get('col')
                transformed_column = pd.Series([np.nan] * X.shape[0], name=column)

                x_values = X[column].unique().tolist()
                print('#x_values: ' + column)
                #                 print(x_values)
                for val in x_values:
                    if val in switch.get('mapping'):
                        if switch.get('mapping')[val]['count'] == 1:
                            transformed_column.loc[X[column] == val] = self._mean
                        else:
                            transformed_column.loc[X[column] == val] = switch.get('mapping')[val]['smoothing']

                if impute_missing:
                    if handle_unknown == 'impute':
                        transformed_column.fillna(self._mean, inplace=True)
                    elif handle_unknown == 'error':
                        missing = transformed_column.isnull()
                        if any(missing):
                            raise ValueError('Unexpected categories found in column %s' % switch.get('col'))

                X[column] = transformed_column.astype(float)

        else:
            self._mean = y.mean()
            prior = self._mean
            mapping_out = []
            for col in cols:
                tmp = y.groupby(X[col]).agg(['sum', 'count'])
                tmp['mean'] = tmp['sum'] / tmp['count']
                tmp = tmp.to_dict(orient='index')

                for val in tmp:
                    smoothing = smoothing_in
                    smoothing = 1 / (1 + np.exp(-(tmp[val]["count"] - min_samples_leaf) / smoothing))
                    cust_smoothing = prior * (1 - smoothing) + tmp[val]['mean'] * smoothing
                    tmp[val]['smoothing'] = cust_smoothing

                mapping_out.append({'col': col, 'mapping': tmp}, )

        return X, mapping_out

class EnhancedLeaveOneOutEncoder(LeaveOneOutEncoder):

    def transform_leave_one_out(self, X_in, y, mapping=None, impute_missing=True, handle_unknown='impute'):
        """
        Leave one out encoding uses a single column of floats to represent the means of the target variables.
        """

        X = X_in.copy(deep=True)

        random_state_ = check_random_state(self.random_state)
        for switch in mapping:
            column = switch.get('col')
            transformed_column = pd.Series([np.nan] * X.shape[0], name=column)

            x_values = set(X[column])
            val_mapping = switch.get('mapping')
            
            
            for val in x_values:
                if val in val_mapping:
                    if y is None:
                        transformed_column.loc[X[column] == val] = val_mapping[val]['mean']
                    #                         print("... val transform done: " + val)
                    elif val_mapping[val]['count'] == 1:
                        transformed_column.loc[X[column] == val] = self._mean
                    else:
                        transformed_column.loc[X[column] == val] = (
                            (val_mapping[val]['sum'] - y[(X[column] == val).values]) / (
                                val_mapping[val]['count'] - 1)
                        )

            if impute_missing:
                if handle_unknown == 'impute':
                    transformed_column.fillna(self._mean, inplace=True)
                elif handle_unknown == 'error':
                    missing = transformed_column.isnull()
                    if any(missing):
                        raise ValueError('Unexpected categories found in column %s' % column)

            if self.randomized and y is not None:
                transformed_column = (transformed_column * random_state_.normal(1., self.sigma, transformed_column.shape[0]))

            X[column] = transformed_column.astype(float)
        return X
