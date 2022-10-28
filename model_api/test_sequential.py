from sklearn.model_selection import KFold
from tensorflow import convert_to_tensor as ctt

def train_and_test_CV_tf(model, X, y, k=10):
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

    score = pearsonr(y_test_tf, curr_y_pred)
    scores.append(score[0])
  
  return y_pred, np.mean(scores)