import numpy as np
import os
import matplotlib.pyplot as plt
#import tensorflow as tf
import plotly.graph_objs as go
from plotly.offline import iplot
import csv

dir = '../datasets3D/banana_1/'
filename1 ='banana_1_2_15.pcd'

x=[]
y=[]
z=[]

with open(os.path.join(dir, filename1),'r') as f:

    reader = csv.reader(f,delimiter=' ')
    next(reader) #0
    next(reader) #1
    next(reader) #2
    next(reader) #3
    next(reader) #4
    next(reader) #5
    next(reader) #6
    next(reader) #7
    next(reader) #8
    next(reader) #9
    for row in reader:
        x.append(row[0])
        y.append(row[1])
        z.append(row[2])


x_c = [i for i in x]
y_c = [i for i in y]
z_c = [i for i in z]


trace = go.Scatter3d(x=x_c, y=y_c, z=z_c, mode='markers', 
                      marker=dict(size=2, color=(0,125,125), colorscale='Viridis', opacity=0.7))
data = [trace]
layout = go.Layout(height=1000, width=1000, title = " in 3D space")
fig = go.Figure(data=data, layout=layout)
iplot(fig)