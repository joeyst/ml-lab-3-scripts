from sklearn.model_selection import KFold
from tensorflow import convert_to_tensor as ctt
import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf

def train_and_test_CV_tf(model, X, y, k=10, summary=False, comparison=False, epochs=5, batch_size=100):
  kfold = KFold(n_splits=k).split(X)

  scores = []
  y_pred = []

  X = X.values
  y = y.values

  for train, test in kfold:
    temp_model = tf.keras.models.clone_model(model)

    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    X_train = np.asarray(X_train).astype('float32')
    X_test  = np.asarray(X_test).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    y_test  = np.asarray(y_test).astype('float32')

    temp_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    curr_y_pred = temp_model.predict(X_test)

    if comparison:
      print("Pred:", curr_y_pred)
      print("Act :", y_test)

    y_pred.extend(curr_y_pred)

    score = pearsonr(y_test, curr_y_pred)
    scores.append(score[0])
  
  if summary:
    print(temp_model.summary())

  return y_pred, np.mean(scores)