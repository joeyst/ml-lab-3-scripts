from sklearn.model_selection import KFold
from tensorflow import convert_to_tensor as ctt
import numpy as np
from scipy.stats import pearsonr

def train_and_test_CV_tf(model, X, y, k=10, verbose=False):
  kfold = KFold(n_splits=k).split(X)

  scores = []
  y_pred = []

  X = np.asarray(X.values).astype('float32')
  y = np.asarray(y.values).astype('float32')

  for train, test in kfold:
    X_train, X_test = X, X
    y_train, y_test = y, y

    print("X_train_tf:", X_train.shape)
    print("y_train_tf:", y_train.shape)

    model.fit(X_train_tf, y_train)
    curr_y_pred = model.predict(X_test)

    print("X_test:", X_test.shape)
    print("y_pred:", curr_y_pred.shape)

    y_pred.extend(curr_y_pred)

    score = pearsonr(y_test, curr_y_pred)
    scores.append(score[0])
  
  if verbose:
    print(model.summary())

  return y_pred, np.mean(scores)