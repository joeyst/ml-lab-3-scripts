import keras as k
import random

def get_Dense():
  return k.layers.Dense(random.randint(2, 20))

def get_Dropout():
  return k.layers.Dropout(random.randint(10, 100) / 100)

def get_Activation():
  return k.layers.Activation(k.activations.relu)

def get_Layer():
  return random.choice([get_Dense(), get_Dropout(), get_Activation()])

def get_Sequential(X_df, layers=3):
  model = k.Sequential()

  model.add(k.Input(shape=len(X_df.columns)))
  for _ in range(layers):
    model.add(get_Layer())
  model.add(k.Dense(1, activation='softmax'))

  model.compile()
  return model
