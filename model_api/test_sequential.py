from sklearn.model_selection import KFold
from tensorflow import convert_to_tensor as ctt
import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf

def train_and_test_CV_tf(model, X, y, k=10, summary=False, comparison=False, epochs=5, batch_size=100, opt='adam', loss_fn='mean_squared_error', run_eagerly=False):
  kfold = KFold(n_splits=k).split(X)

  scores = []
  y_pred = []

  print("# features:", len(X.columns))
  X = X.values
  y = y.values

  if comparison:
    tf.keras.utils.plot_model(model, show_shapes=True)

  for train, test in kfold:
    temp_model = tf.keras.models.clone_model(model)
    temp_model.compile(optimizer=opt, loss=loss_fn, run_eagerly=run_eagerly)

    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]

    X_train = tf.convert_to_tensor(np.asarray(X_train).astype('float64'))
    X_test  = tf.convert_to_tensor(np.asarray(X_test).astype('float64'))
    y_train = tf.convert_to_tensor(np.asarray(y_train).astype('float64'))
    y_test  = np.asarray(y_test).astype('float64')

    print("X_train shape:", X_train.shape)
    print("Y_train shape:", y_train.shape)
    history = temp_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    curr_y_pred = temp_model.predict(X_test)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", y_test.shape)

    if comparison:
      print("Pred:", curr_y_pred)
      print("Act :", y_test)

    y_pred.extend(curr_y_pred)

    score = pearsonr(y_test, curr_y_pred)
    scores.append(score[0])
  
  if summary:
    print(temp_model.summary())

  return y_pred, np.mean(scores)
