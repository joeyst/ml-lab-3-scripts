from sklearn.model_selection import KFold
from tensorflow import convert_to_tensor as ctt
import numpy as np
from scipy.stats import pearsonr

def train_and_test_CV_tf(model, X, y, k=10, verbose=False):
  kfold = KFold(n_splits=k).split(X)

  scores = []
  y_pred = []

  X = X.values
  y = y.values

  for train, test in kfold:
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    X_train = np.asarray(X_train).astype('float32')
    X_test  = np.asarray(X_test).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    y_test  = np.asarray(y_test).astype('float32')

    print("X_train_tf:", X_train.shape)
    print("y_train_tf:", y_train.shape)

    model.fit(X_train, y_train)
    curr_y_pred = model.predict(X_test)

    print("y_test:", y_test.shape)
    print("y_test type:", type(y_test))
    print("y_pred:", curr_y_pred.shape)
    print("y_pred type:", type(curr_y_pred))

    y_pred.extend(curr_y_pred)

    score = pearsonr(y_test, curr_y_pred)
    scores.append(score[0])
  
  if verbose:
    print(model.summary())

  return flatten(y_pred), np.mean(scores)