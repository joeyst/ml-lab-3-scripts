import keras as k
import random

def get_Dense():
  return k.layers.Dense(random.randint(2, 20))

def get_Dropout():
  return k.layers.Dropout(random.randint(0, 50) / 100)

def get_Activation():
  return k.layers.Activation(k.activations.relu)

def get_Layer():
  return random.choice([get_Dense(), get_Dropout(), get_Activation()])

def get_random_Sequential(X_df, layers=3, opt='adam', loss_fn='mean_squared_error'):
  model = k.Sequential()

  model.add(k.Input(shape=len(X_df.columns)))
  for _ in range(layers):
    model.add(get_Layer())
  model.add(k.layers.Dense(1))
  model.add(k.layers.Lambda(lambda x : (x - 0.5) * 2))

  model.compile(optimizer=opt, loss=loss_fn)
  return model
