from sklearn.model_selection import KFold
from tensorflow import convert_to_tensor as ctt
import numpy as np
from scipy.stats import pearsonr

def train_and_test_CV_tf(model, X, y, k=10, verbose=True):
  kfold = KFold(n_splits=k).split(X)

  scores = []
  y_pred = []

  X = np.asarray(X.values).astype('float32')
  y = np.asarray(y.values).astype('float32')

  for train, test in kfold:
    X_train_tf, X_test_tf = ctt(X[train]), ctt(X[test])
    y_train_tf, y_test_tf = ctt(y[train]), ctt(y[test])
    y_test = y[test]

    model.fit(X_train_tf, y_train_tf)
    curr_y_pred = model.predict(X_test_tf)

    y_pred.extend(curr_y_pred)

    print("X_train_tf:", X_train_tf)
    print("Type:", type(X_train_tf))
    print("\ny_pred:", y_pred)
    print("Type:", type(y_pred))
    print("\ny_test:", y_test)
    print("Type:", type(y_test))
    score = pearsonr(y_test, curr_y_pred)
    scores.append(score[0])
  
  if verbose:
    print(model.summary())

  return y_pred, np.mean(scores)