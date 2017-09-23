import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from sklearn import decomposition

################################################################################
### AscendingOrderer                                                         ###
################################################################################
class Ascending_order(BaseEstimator, TransformerMixin):
  def __init__(self, columns):
    self.columns = columns

  def fit(self, df, y=None):
    return self

  def transform(self, df, y=None):
    df2 = df.sort_values(by=self.columns)
    return df2


################################################################################
### ColumnDropper                                                            ###
################################################################################
class Columns_dropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df2 = df.drop(self.columns, 1)
        return df2


################################################################################
### Divider                                                                  ###
################################################################################
class Divider(BaseEstimator, TransformerMixin):
    def __init__(self, column_1, column_2, column_result):
        self.column_1 = column_1
        self.column_2 = column_2
        self.column_result = column_result

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df[self.column_result] = df[self.column_1] / df[self.column_2]
        return df


################################################################################
### Normalizer                                                               ###
################################################################################
class Normalizer(BaseEstimator, TransformerMixin):
    def __init__(self,):
        return

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df = (df - df.min()) / (df.max() - df.min())
        return df


################################################################################
### OneHotEncoder                                                            ###
################################################################################

class One_hot_enconder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        encoder = preprocessing.LabelEncoder()
        encoder2 = df[[self.columns]].apply(encoder.fit_transform)
        df[self.columns] = encoder2
        return df


################################################################################
### Substractor                                                              ###
################################################################################
class Substractor(BaseEstimator, TransformerMixin):
  def __init__(self, column_1, column_2, column_result):
    self.column_1 = column_1
    self.column_2 = column_2
    self.column_result = column_result

  def fit(self, df, y=None):
    return self

  def transform(self, df, y=None):
    df[self.column_result] = df[self.column_1] - df[self.column_2]
    return df

################################################################################
### Adder                                                                    ###
################################################################################
class Adder(BaseEstimator, TransformerMixin):
  def __init__(self, column_1, column_2, column_result):
    self.column_1 = column_1
    self.column_2 = column_2
    self.column_result = column_result

  def fit(self, df, y=None):
    return self

  def transform(self, df, y=None):
    df[self.column_result] = df[self.column_1] + df[self.column_2]
    return df

################################################################################
### Multiplier                                                               ###
################################################################################
class Multiplier(BaseEstimator, TransformerMixin):
  def __init__(self, column_1, column_2, column_result):
    self.column_1 = column_1
    self.column_2 = column_2
    self.column_result = column_result

  def fit(self, df, y=None):
    return self

  def transform(self, df, y=None):
    df[self.column_result] = df[self.column_1] * df[self.column_2]
    return df


################################################################################
### FilterValues                                                             ###
################################################################################
class Filter_values(BaseEstimator, TransformerMixin):
    def __init__(
            self, column, value_min=None, value_max=None, type_inter=True
    ):
        self.column = column
        self.value_min = value_min
        self.value_max = value_max
        self.type_inter = type_inter

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        if ((self.value_min is None) & (self.value_max is None)):
            return df

        elif (self.value_max is not None):
            if self.type_inter:
                return df[df[self.column] <= self.value_max]
            else:
                return df[df[self.column] < self.value_max]

        elif (self.value_min is not None):
            if self.type_inter:
                return df[df[self.column] >= self.value_min]
            else:
                return df[df[self.column] > self.value_min]

        else:
            if self.type_inter:
                return df[(df[self.column] >= self.value_min) & (df[self.column] <= self.value_max)]
            else:
                return df[(df[self.column] > self.value_min) & (df[self.column] < self.value_max)]

################################################################################
### ChangeValue                                                              ###
################################################################################
class Change_value(BaseEstimator, TransformerMixin):
  def __init__(self, columns, value_initial=None, value_change=None):
    self.columns = columns
    self.value_change = value_change
    self.value_initial = value_initial

  def fit(self, df, y=None):
    return self

  def transform(self, df, y=None):
    df.loc[df[self.columns] == self.value_initial, self.columns] = self.value_change
    return df


################################################################################
### SetTime                                                                  ###
################################################################################
class Set_time(BaseEstimator, TransformerMixin):
  def __init__(self, columns, frequency):
    self.columns = columns
    self.frequency = frequency

  def fit(self, df, y=None):
    return self

  def transform(self, df, y=None):
      aju = df.reset_index()
      aju[self.columns] = aju[self.columns].dt.floor(self.frequency)
      return aju


################################################################################
### Decomposition                                                            ###
################################################################################
class Decomposition(BaseEstimator, TransformerMixin):
  def __init__(self, n_components=2, random_state=420):
    self.n_components = n_components
    self.random_state = random_state

  def fit(self, df, y=None):
    return self

  def transform(self, df, y=None):
      pca = decomposition.PCA(n_components=self.n_components, random_state=self.random_state)
      pca_train = pca.fit_transform(df)

      ica = decomposition.FastICA(n_components=self.n_components, random_state=self.random_state)
      ica_train = ica.fit_transform(df)

      tsvd = decomposition.TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
      tsvd_train = tsvd.fit_transform(df)

      nmf = decomposition.NMF(n_components=self.n_components, random_state=self.random_state)
      nmf_train = nmf.fit_transform(df)

      for i in range(1, self.n_components + 1):
          df['pca_' + str(i)] = pca_train[:, i - 1]

          df['ica_' + str(i)] = ica_train[:, i - 1]

          df['tsvd_' + str(i)] = tsvd_train[:, i - 1]

          df['nmf_' + str(i)] = nmf_train[:, i - 1]

      return df


