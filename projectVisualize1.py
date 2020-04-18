import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
#import tensorflow as tf
import plotly.graph_objs as go
from plotly.offline import iplot

dir = '../datasets3D/3D_MNIST/'
filename ='train_point_clouds.h5'

index = 20

with h5py.File(os.path.join(dir, filename), "r") as points_dataset:
    digits = []
    for i in range(0,200):
        digit = (points_dataset[str(i)]["img"][:], 
                 points_dataset[str(i)]["points"][:], 
                 points_dataset[str(i)].attrs["label"])
        digits.append(digit)
x_c = [r[0] for r in digits[index][1]]
y_c = [r[1] for r in digits[index][1]]
z_c = [r[2] for r in digits[index][1]]

trace = go.Scatter3d(x=x_c, y=y_c, z=z_c, mode='markers', 
                      marker=dict(size=1, color=z_c, colorscale='Viridis', opacity=0.7))
data = [trace]
layout = go.Layout(height=500, width=600, title= "Digit: "+str(digits[index][2]) + " in 3D space")
fig = go.Figure(data=data, layout=layout)
iplot(fig)