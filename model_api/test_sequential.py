from sklearn.model_selection import KFold
from tensorflow import convert_to_tensor as ctt

def train_and_test_CV_tf(model, X, y, k=10):
  kfold = KFold(n_splits=k).split(X)

  scores = []
  y_pred = []

  for train, test in kfold:
    X_train_tf, X_test_tf = ctt(X.values[train]), ctt(X.values[test])
    y_train_tf, y_test_tf = ctt(y.values[train]), ctt(y.values[test])
    y_test = y.values[test]

    model.fit(X_train, y_train)
    curr_y_pred = model.predict(X_test)

    y_pred.extend(curr_y_pred)

    score = pearsonr(y_test, curr_y_pred)
    scores.append(score[0])
  
  return y_pred, np.mean(scores)