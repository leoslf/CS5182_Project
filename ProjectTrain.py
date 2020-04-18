import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

dir = '../datasets3D/3D_MNIST/'
filename ='full_dataset_vectors.h5'

def array_to_color(array, cmap="Oranges"):
  s_m = plt.cm.ScalarMappable(cmap=cmap)
  return s_m.to_rgba(array)[:,:-1]

def rgb_data_transform(data):
  data_t = []
  for i in range(data.shape[0]):
    data_t.append(array_to_color(data[i]).reshape(16, 16, 16, 3))
  return np.asarray(data_t, dtype=np.float32)


with h5py.File(os.path.join(dir, filename), "r") as hf:    

    # Split the data into training/test features/targets
    x_train = hf["X_train"][:]
    y_train = hf["y_train"][:]
    x_test = hf["X_test"][:] 
    y_test = hf["y_test"][:]

#print("x_train shape", x_train.shape)
#print("y_train shape", y_train.shape)
#print("x_test shape", x_test.shape)
#print("y_test shape", y_test.shape)

print (x_train.shape)
print (y_train.shape)

#Initialising the channel dimension in the input dataset
xtrain = np.ndarray((x_train.shape[0], 4096, 3))
xtest = np.ndarray((x_test.shape[0], 4096, 3))

def add_rgb_dimen(array):
    scalar_map = plt.cm.ScalarMappable(cmap="Oranges")
    array = scalar_map.to_rgba(array)[:,:-1]
    return array

for i in range(x_train.shape[0]):
    xtrain[i] = add_rgb_dimen(x_train[i])

for i in range(x_test.shape[0]):
    xtest[i] = add_rgb_dimen(x_test[i])

#convert to 4D space
xtrain = xtrain.reshape(x_train.shape[0], 16, 16, 16, 3)
xtest = xtest.reshape(x_test.shape[0], 16, 16, 16, 3)

#one hot for target variable
ytrain = to_categorical(y_train, 10)
ytest = to_categorical(y_test, 10)

#print("xtrain shape = ", xtrain.shape)
#print("ytrain shape = ", ytrain.shape)
   
#print(ytrain[:10])

from projectModel import *

model = model_v3((16,16,16,3), classQTY = 10)

model.summary()

monitor = EarlyStopping(monitor='val_loss', min_delta =0.001, patience = 15, verbose = 0,
                       mode = 'auto', restore_best_weights = True )

history = model.fit(x=xtrain, 
                    y=ytrain,
                    batch_size=256, 
                    epochs=1000, 
                    validation_split=0.2, 
                    callbacks=[monitor])


model.save("simpleCNN_t3_v3.h5")
df = pd.DataFrame(history.history)
filename = 'history_simpleCNN_t3_v3.csv'
with open (filename, mode ='w') as f:
    df.to_csv(f)
