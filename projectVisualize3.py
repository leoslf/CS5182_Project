import numpy as np
import os
import sys
import csv
import pickle
import plotly.graph_objs as go
from plotly.offline import iplot
import matplotlib.cm
import matplotlib.pyplot as plt

dir = '../datasets3D/PointData/toilet/train/'
fname = 'toilet_0221.off'
x= []
y=[]
z=[]
def readFile (dir, fname):
    with open(os.path.join(dir,fname)) as f:
        content = f.readlines()
        n_points = int (content[1].split()[0])
        print (n_points)
        #points = content[2:n_points]
        for i in range (n_points):
            x.append(content[i+2].split()[0])
            y.append(content[i+2].split()[1])
            z.append(content[i+2].split()[2])
    return x, y, z

x, y, z = readFile (dir, fname)

classes = ['chair', 'desk', 'monitor'
           ,'night_stand','table', 'toilet']

index = 3

trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', 
                      marker=dict(size=1, color = (255,0,0) , colorscale='Viridis', opacity=0.7))
data = [trace]
layout = go.Layout(height=500, width=500, 
                   title= "Actual: " + str (classes[index]))
fig = go.Figure(data=data, layout=layout)
iplot(fig)
