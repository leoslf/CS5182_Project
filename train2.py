import argparse
import os
import sys
import pickle
from datetime import datetime
from random import shuffle
from data_utils import *
from prepareData import *
import plotly.graph_objs as go
from plotly.offline import iplot
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

dir = '../datasets3D/PointData/'
threshold = 1944    #1944 x 3= 5832
#threshold = 3087    #3087 x 3  (no enough memory)

#prepare_datasets(dir, threshold)

data = pickle.load(open('data.pickle', "rb" ))

train_list = data['train_list']
eval_list = data['eval_list']
test_list = data['test_list']
class_dict = data['class_dict']

x_train, y_train = get_points_and_class(train_list, class_dict, threshold, False )

x_train = x_train.reshape(x_train.shape[0],-1)

print (x_train.shape)
print (y_train.shape)

def array_to_color(array, cmap="Oranges"):
  s_m = plt.cm.ScalarMappable(cmap=cmap)
  return s_m.to_rgba(array)[:,:-1]

def rgb_data_transform(data):
  data_t = []
  for i in range(data.shape[0]):
    data_t.append(array_to_color(data[i]).reshape(18, 18, 18, 3))
  return np.asarray(data_t, dtype=np.float32)

def add_rgb_dimen(array):
    scalar_map = plt.cm.ScalarMappable(cmap="Oranges")
    array = scalar_map.to_rgba(array)[:,:-1]
    return array

#Initialising the channel dimension in the input dataset
xtrain = np.ndarray((x_train.shape[0], 5832, 3))


for i in range(x_train.shape[0]):
    xtrain[i] = add_rgb_dimen(x_train[i])

#convert to 4D space
xtrain = xtrain.reshape(x_train.shape[0], 18, 18, 18, 3)
#print (xtrain[0])
ytrain = to_categorical(y_train, 4)

from projectModel import *

model = model_v5((18,18,18,3), 4)

model.summary()

monitor = EarlyStopping(monitor='val_loss', min_delta =0.001, patience = 50, verbose = 0,
                       mode = 'auto', restore_best_weights = True )

history = model.fit(x=xtrain, 
                    y=ytrain,
                    batch_size=256, 
                    epochs=1000, 
                    validation_split=0.2, 
                    callbacks=[monitor])


model.save("simpleCNN_v7.h5")
df = pd.DataFrame(history.history)
filename = 'history_simpleCNN_v7.csv'
with open (filename, mode ='w') as f:
    df.to_csv(f)