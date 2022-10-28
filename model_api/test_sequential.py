from sklearn.model_selection import KFold
from tensorflow import convert_to_tensor as ctt
import numpy as np
from scipy.stats import pearsonr

def train_and_test_CV_tf(model, X, y, k=10, summary=False, comparison=False):
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

    model.fit(X_train, y_train)
    curr_y_pred = model.predict(X_test)

    if comparison:
      print("Pred:", curr_y_pred)
      print("Act :", y_test)

    y_pred.extend(curr_y_pred)

    score = pearsonr(y_test, curr_y_pred)
    scores.append(score[0])
  
  if summary:
    print(model.summary())

  return y_pred, np.mean(scores)