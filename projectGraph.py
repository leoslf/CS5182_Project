import numpy as np
import tensorflow as tf
from tensorflow import keras
import pydotplus
from IPython.display import SVG
import h5py
import pandas as pd
import os
import matplotlib.pyplot as plt


#modelDir = ""
#loaded_model = ".h5"
#model = tf.keras.models.load_model(os.path.join(modelDir, loaded_model))

#model.summary()
#tf.keras.utils.vis_utils.pydot = pydot
#tf.keras.utils.plot_model(
#    model, to_file=os.path.join(dir, "t15.png") , show_shapes=False, show_layer_names=True,
#    rankdir='TB', expand_nested=False, dpi=96)

dir = ''
filename = 'history_simpleCNN_v4.csv'
trainingTime = 30


df = pd.read_csv(os.path.join(dir, filename))

x1 = df['accuracy']
x2 = df['val_accuracy']
x3 = df['loss']
x4 = df['val_loss']

def plot_learningCurve(x1, x2, x3, x4, epoch):
    #fig, axes = plt.subplots (1, 2)
    plt.figure(1)
    epoch_range = range(1,epoch+1)
    plt.plot(epoch_range, x1)
    plt.plot(epoch_range, x2)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train- accuracy', 'Val-accuracy'], loc = 'upper left')
    #plt.show()

    plt.figure(2)
    plt.plot(epoch_range, x3)
    plt.plot(epoch_range, x4)
    plt.title ('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train-loss', 'Val-loss'], loc = 'upper right')

    plt.tight_layout()
    plt.show()

plot_learningCurve(x1, x2, x3, x4, trainingTime)
