import numpy as np
import h5py
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import iplot
from tensorflow.keras.utils import to_categorical

# This program output the accuracy, and 0-30 predicted test, and the ground true
# model and weight is on line 14.  It can be replaced. 

dir = "./weight/"
loaded_model = "simpleCNN_t5_v3.h5"
model = tf.keras.models.load_model(os.path.join(dir+loaded_model))

test_data_dir = '../datasets3D/3D_MNIST/'
filename ='full_dataset_vectors.h5'

def add_rgb_dimen(array):
    scalar_map = plt.cm.ScalarMappable(cmap="Oranges")
    array = scalar_map.to_rgba(array)[:,:-1]
    return array


with h5py.File(os.path.join(test_data_dir, filename), "r") as hf:    

    x_test = hf["X_test"][:] 
    y_test = hf["y_test"][:]


xtest = np.ndarray((x_test.shape[0], 4096, 3))

for i in range(x_test.shape[0]):
    xtest[i] = add_rgb_dimen(x_test[i])

xtest = xtest.reshape(x_test.shape[0], 16, 16, 16, 3)
ytest = to_categorical(y_test, 10)

prediction = model.predict(xtest)
prediction = np.argmax(prediction, axis = 1)
print("------------------------------------------------------------")
# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(y_test, prediction)])
accuracy = match_count / len(prediction)
print()
print("Accuracy: {:.3f}".format(accuracy))
print()
print ("Result in test set for first 30 samples")
print ("Prediction:", prediction[0:30])
print ("Actual    :", np.argmax(ytest[0:30], axis=1))

