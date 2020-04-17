import numpy as np
import os
import sys
import csv
import pickle
from data_utils import *
from prepareData import *
import plotly.graph_objs as go
from plotly.offline import iplot

dir = '../datasets3D/PointData/'
threshold = 1944    #1944 x 3= 5832

#prepare_datasets(dir, threshold)

data = pickle.load(open('data.pickle', "rb" ))

train_list = data['train_list']
eval_list = data['eval_list']
test_list = data['test_list']
class_dict = data['class_dict']

x_train, y_train = get_points_and_class(train_list, class_dict, threshold, False )

index = 6
#print (x_train[index][:10])

x_c = [r[0] for r in x_train[index]]
y_c = [r[1] for r in x_train[index]]
z_c = [r[2] for r in x_train[index]]

print (y_train[index])
classes = ['chair', 'desk', 'night_stand', 'toilet']


trace = go.Scatter3d(x=x_c, y=y_c, z=z_c, mode='markers', 
                      marker=dict(size=2, color= (255,0,0) , colorscale='Viridis', opacity=0.7))
data = [trace]
layout = go.Layout(height=500, width=500, 
                   title= str(classes[y_train[index]]) + " in 3D space")
fig = go.Figure(data=data, layout=layout)
iplot(fig)