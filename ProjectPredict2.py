import numpy as np
import argparse
import os
import sys
import pickle
from data_utils import *
from prepareData import *
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import to_categorical
import plotly.graph_objs as go
from plotly.offline import iplot

dir = '../datasets3D/PointData/'
threshold = 1944    #1944 x 3= 5832
#threshold = 3087    #3087 x 3  (no enough memory)

fname = 'data.pickle'

data = pickle.load(open(fname, "rb" ))

train_list = data['train_list']
eval_list = data['eval_list']
test_list = data['test_list']
class_dict = data['class_dict']


#print (class_dict)

#for i in range (len(class_dict)):
#    classes[i] = class_dict[i]

classes = ['chair', 'desk', 'monitor'
           ,'night_stand','table', 'toilet']

x_test, y_test = get_points_and_class(test_list, class_dict, threshold, False )

index = 3
print (test_list[index])

x_c = [r[0] for r in x_test[index]]
y_c = [r[1] for r in x_test[index]]
z_c = [r[2] for r in x_test[index]]

x_test = x_test.reshape(x_test.shape[0],-1)

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
xtest = np.ndarray((x_test.shape[0], 5832, 3))


for i in range(x_test.shape[0]):
    xtest[i] = add_rgb_dimen(x_test[i])

#convert to 4D space
xtest = xtest.reshape(x_test.shape[0], 18, 18, 18, 3)
#print (xtrain[0])
ytest = to_categorical(y_test, len(classes))

from projectModel import *

dir = "./weight/"
loaded_model = "simpleCNN_t8_v3.h5"
model = tf.keras.models.load_model(os.path.join(dir+loaded_model))

prediction = model.predict(xtest)
prediction = np.argmax(prediction, axis = 1)

print (xtest.shape)

# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(y_test, prediction)])
accuracy = match_count / len(prediction)
print("Accuracy: {:.3f}".format(accuracy))

print ("Prediction:", prediction[:30])
print ("Actual    :", np.argmax(ytest[:30], axis=1))
print ("Pre: ", prediction[index])


trace = go.Scatter3d(x=x_c, y=y_c, z=z_c, mode='markers', 
                      marker=dict(size=2, color= (255,0,0) , colorscale='Viridis', opacity=0.7))
data = [trace]
layout = go.Layout(height=500, width=500, 
                   title= "Predict: " + str(classes[prediction[index]]) + " / " + "Actual: " 
                   + str (classes[y_test[index]]))
fig = go.Figure(data=data, layout=layout)
iplot(fig)


