import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from scipy.stats import pearsonr
from scipy.special import kl_div

class DataVisualizer:
  """
  __init__ : `X` => DataFrame or Series, `y` => Series
  scatter  : creates scatterplot with line showing relationship between each feature and the label 
  hist     : creates histogram of each feature and/or the label 

  pipeline : allows transformation of data and plotting 
  transform: transforms specified features and/or label 
  reset    : resets data to untransformed state 

  get_transform_history: gets columns, transforms applied (or resets), and whether label was included 
  """

  def __init__(self, X_frame, y_frame):
    self._X_frame = X_frame
    self._y_frame = y_frame
    self._cleanup_data()
    self.X_state = self._X_frame
    self.y_state = self._y_frame
    self.transform_history = []

  def pipeline(self, visualizer='scatter', transforms=[], feats=None, label=False, normalize=True, make_nonnegative=True, **kwargs):
    # apply transforms 
    self.transform(transforms=transforms, feats=feats, label=label, normalize=normalize, make_nonnegative=make_nonnegative)

    # visualize 
    if visualizer == 'scatter':
      return self.scatter(feats=feats, **kwargs)
    #elif visualizer == 'hist':


  def reset(self):
    self.X_state = self._X_frame
    self.y_state = self._y_frame
    self._append_transform_history(transform="reset")

  def _make_nonnegative(self, df, verbose=False):
    if isinstance(df, pd.Series):
      if verbose:
        print("`_make_nonnegative`: passed in `Series` object; converting to `DataFrame`")

      df = df.to_frame()
    elif not isinstance(df, pd.DataFrame):
      raise Exception("`_make_nonnegative`: passed in {} object; not `Series` or `DataFrame`".format(type(df)))

    for (name, data) in df.iteritems():
      if data.min() < 0:
        nonzeromax = MinMaxScaler(feature_range=(0, data.max()))
        df.update(pd.DataFrame(nonzeromax.fit_transform(data.to_numpy().reshape(-1, 1)), columns=[name]))

    return df

  def transform(self, transforms, feats=None, label=False, normalize=True, make_nonnegative=True):
    # get `DataFrame` of desired features 
    features = self._select_columns(feats)

    # conditionally ensure values are nonnegative 
    if make_nonnegative:
      features = self._make_nonnegative(features)

    # get `Series` of label
    labels = self._get_y()

    if not transforms == None and not isinstance(transforms, list) and not transforms == []:
      transforms = [transforms]

    # transform desired features 
    if not len(transforms) == 0:
      features = features.transform(transforms)

    # transform label if desired 
    if label and not len(transforms) == 0:
      labels = labels.transform(transforms)

    if normalize:
      # get normalizer 
      minmax = MinMaxScaler(feature_range=(0, 1))

      # transform features 
      cols = features.columns
      X_transformed = pd.DataFrame(minmax.fit_transform(features))
      X_transformed.columns = cols      

    # update `X_state` 
    self._update_X(X_transformed)

    # update `y_state`
    self._update_y(labels)

    # record transform and columns affected 
    self._append_transform_history(columns=feats, transform=transforms, label_included=label)

  def _update_X(self, frame):
    curr_X = self._get_X()
    curr_X.update(frame)
    self._set_X(curr_X)

  def _update_y(self, series):
    curr_y = self._get_y()
    curr_y.update(series)
    self._set_y(curr_y)

  def _get_X(self):
    return self.X_state

  def _get_y(self):
    return self.y_state

  def _set_X(self, frame):
    self.X_state = frame

  def _set_y(self, frame):
    self.y_state = frame

  def _append_transform_history(self, **data):
    self.transform_history.append(data)

  def get_transform_history(self):
    return self.transform_history

  def _cleanup_data(self):
    """ ensure datatypes are correct and coerce to `float64` """
    self._ensure_dataframes()
    self._coerce_to_float()
    
  def _ensure_dataframes(self):
    """ ensure datatypes are correct and convert from `Series` to `DataFrame` if needed """
    # ensure good inputs 
    if not isinstance(self._X_frame, (pd.DataFrame, pd.Series)):
      raise Exception("`display_correlation`: `df_features` is not a `DataFrame`")
    if not isinstance(self._y_frame, pd.Series):
      raise Exception("`display_correlation`: `df_labels` is not a `DataFrame`")

    # ensure `X` is a `DataFrame`
    if isinstance(self._X_frame, pd.Series):
      self._X_frame = self._X_frame.to_frame()

  def _coerce_to_float(self):
    """ coerce data to `float64` """
    self._X_frame = self._X_frame.astype('float64')
    self._y_frame = self._y_frame.astype('float64')

  def _apply_to_each_feature_with_label(self, function, df, verbose=False):
    """ 
    takes function that takes in two lists of values 
    and applies to each `(feature, label)` pair  
    """

    if isinstance(df, pd.Series):
      if verbose:
        print("`_make_nonnegative`: passed in `Series` object; converting to `DataFrame`")

      df = df.to_frame()
    elif not isinstance(df, pd.DataFrame):
      raise Exception("`_make_nonnegative`: passed in {} object; not `Series` or `DataFrame`".format(type(df)))

    output = {}

    for (name, data) in df.iteritems():
      output[name] = function(self._get_y(), data)
    
    return output

  def pearsonr(self, feats=None):
    df = self._select_columns(feats)
    return self._apply_to_each_feature_with_label(pearson_corr, df)

  def kl_div(self, feats=None):
    df = self._select_columns(feats)
    df = self._make_nonnegative(df)
    return self._apply_to_each_feature_with_label(kl_div, df)

  def hist(self, feats=None, label=True, kbins=10, **kwargs):
    """
    feats: list | string | None => which feature(s) to display 
    kbins: integer 
    """

    # get current `DataFrame`s
    curr = self._select_columns(feats)

    if label:
      curr = pd.concat([curr, self._get_y().to_frame()], axis=1)

    # set up plot 
    fig, ax = plt.subplots(figsize=(18, 10))

    # plot data 
    ax.hist(curr.values, kbins, histtype='step', stacked=False, fill=False, label=curr.columns, linewidth=7, **kwargs)

    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    plt.xlabel('Range of label values')
    plt.ylabel('# of label values in bin')
    plt.title('Frequency of datapoints')
    plt.legend(loc='upper left')
    plt.show()

    return self.kl_div(feats)

  def scatter(self, feats=None, scat=True, line=True):
    """
    scat : bool => show datapoints
    line : bool => show line
    feats: list | string | None => which feature(s) to display 
      None  : show all features 
      list  : show features with column names matching list items 
      string: show feature with column name matching string
    """

    # get current `DataFrame`s
    curr = self._select_columns(feats)

    # create dictionary to store correlations
    corrs = {}

    # set up plot 
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # plot each feature's correlation with the label 
    for (name, feat_data) in curr.iteritems():
      if scat:
        ax.scatter(self._get_y(), feat_data)

      if line:
        m, b = np.polyfit(self._get_y(), feat_data, 1)
        ax.plot(m*self._get_y() + b, self._get_y(), linewidth=5)

    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    plt.rcParams.update({'font.size': 22})
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.xlabel('Label')
    plt.ylabel('Feature')
    plt.title('Label vs Feature')
    plt.legend(curr.keys(), prop={'size': 20})

    return self.pearsonr(feats)
  
  def _select_columns(self, feats):
    if feats == None:
      return self._get_X()
    elif isinstance(feats, list):
      return self._get_X()[feats]
    elif isinstance(feats, str):
      return self._get_X()[[feats]]
    else:
      raise "`DataVisualizer._select_columns: `feats` is not a `None`, `list`, or `str`."

def pearson_corr(true, pred):
  return pearsonr(true, pred)[0]