import keras as k
import random
from numpy.random import choice

def get_Dense(rang=(2, 20), act=None):
  return k.layers.Dense(random.choice(range(rang[0], rang[1]+1)), activation=act, kernel_initializer=k.initializers.RandomNormal(stddev=0.3), bias_initializer=k.initializers.RandomNormal(stddev=0.3))

def get_Dropout(rang=(0.1, 0.3)):
  return k.layers.Dropout(random.uniform(rang[0], rang[1]))

def get_Activation(act=k.activations.relu):
  return k.layers.Activation(act)

def get_Layer():
  return random.choice([get_Dense(), get_Dropout(), get_Activation()])

def get_Layer_proba(spec_dict):
  props = get_prop_values(spec_dict)
  try:
    deprob = props['dense']
    drprob = props['dropout']
  except:
    print("spec_dict:", props)

  amin = deprob + drprob
  demin = deprob

  if spec_dict.get('dense') == None:
    spec_dict['dense'] = {}
  if spec_dict.get('dropout') == None:
    spec_dict['dropout'] = {}
  if spec_dict.get('activation') == None:
    spec_dict['activation'] = {}

  rand_prop = random.uniform(0, 1)
  if rand_prop >= amin:
    return get_activation_from_extracted(spec_dict['activation'])
  elif rand_prop >= demin:
    return get_dropout_from_extracted(spec_dict['dropout'])
  else:
    return get_dense_from_extracted(spec_dict['dense'])

def get_random_Sequential(X_df, layers=3, opt='adam', loss_fn='mean_squared_error'):
  model = k.Sequential()

  model.add(k.Input(shape=len(X_df.columns)))
  for _ in range(layers):
    model.add(get_Layer())
  model.add(k.layers.Dense(1))
  model.add(k.layers.Lambda(lambda x : (x - 0.5) * 2))

  model.compile(optimizer=opt, loss=loss_fn)
  return model

def get_Sequential(X_df, layers=3, spec_dict={}, opt='adam', loss_fn='mean_squared_error', starting_dense_number=10):
  """
  Returns `Sequential` with specified number of layers, 
  probabilities of types of `Layer`s, and `Layer` parameters. 

  Input: Once nested (two layer) dictionary of 
            specified layers and arguments. 

  Output: `Sequential` 

  Specified: 
  # probabilities of types: 
    # - if probabilities are below one, fills in rest
    #     of probabilities with 1 - that. If no probabilities
    #     left to be specified, normalizes to 1. 
    # - if probabilities exceed one, normalizes to 1. 
  # node range for layers, other arguments 
  # activation functions for layers 

  # types of layers (may change): `Dense`, `Dropout`, `Activation`
    # `Dense` args: prop, act, nodes 
      # prop  => proportion of layers 
      # act   => activation function 
      # nodes => number or range for layer
    # `Dropout` args: prop, drop 
      # prop => proportion of layers
      # drop => number or range to drop 
    # `Activation` args: prop, type
      # prop => proportion of layers
      # type => activation function, or list 

  ```
  spec_dict = {
    'dense': {
      'prop': None,
      'act': None,
      'nodes': None
    },
    'dropout': {
      'prop': None,
      'drop': None
    },
    'activation': {
      'prop': None,
      'type': None
    }
    ```
  }
  """

  model = k.Sequential()

  model.add(k.layers.Dense(units=starting_dense_number, input_shape=len(X_df.columns), activation='tanh', kernel_initializer=k.initializers.RandomNormal(stddev=0.3), bias_initializer=k.initializers.RandomNormal(stddev=0.3)))
  for _ in range(layers):
    model.add(get_Layer_proba(spec_dict))
  model.add(k.layers.Dense(units=1, activation='sigmoid'))
  model.add(k.layers.Lambda(lambda x : (x - 0.5) * 2))

  model.compile(optimizer=opt, loss=loss_fn)
  return model

def get_prop_values(spec_dict):
  return calc_props_from_extracted(extract_props(spec_dict))

def extract_props(spec_dict):
  props = {
    'dense': None,
    'dropout': None,
    'activation': None
  }

  for layer in ['dense', 'dropout', 'activation']:
    if layer in spec_dict.keys():
      if 'prop' in spec_dict[layer].keys():
        props[layer] = spec_dict[layer]['prop']
  
  return props

def get_activation_from_extracted(activation_dict):
  typ = activation_dict.get('type')
  if typ == None:
    typ = 'tanh'
  return get_Activation(typ)

def get_dropout_from_extracted(dropout_dict):
  rang = dropout_dict.get('range')
  if rang == None:
    rang = (0.1, 0.3)
  elif type(rang) == float or type(rang) == int:
    rang = (rang, rang)
  return get_Dropout(rang)

def get_dense_from_extracted(dense_dict):
  act = dense_dict.get('act', 'relu')
  rang = dense_dict.get('range')
  if rang == None: 
    rang = (10, 20)
  elif type(rang) == int:
    rang = (rang, rang)
  return get_Dense(rang, act)


def calc_props_from_extracted(props_dict):
  number_of_nones = 0
  current_sum     = 0.0
  for layer, prop in props_dict.items():
    if prop == None:
      number_of_nones += 1
    else:
      current_sum += prop

  if number_of_nones == 3:
    return {'dense': 0.8, 'dropout': 0.1, 'activation': 0.1}

  props = {}
  if (current_sum > 1) or (current_sum < 1 and number_of_nones == 0):
    for layer, prop in props_dict.items():
      if prop == None:
        props[layer] = 0
      else:
        props[layer] = prop / current_sum
  elif current_sum < 1:
    default_prop = (1 - current_sum) / number_of_nones
    for layer, prop in props_dict.items():
      if prop == None:
        props[layer] = default_prop
      else:
        props[layer] = prop
  else:
    for layer, prop in props_dict.items():
      if prop == None:
        props[layer] = 0
      else:
        props[layer] = prop
  
  return props