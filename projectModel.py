import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model

def model_v1 (input = (16,16,16,3), classQTY =10, rate = 0.001 ):

    model = Sequential()
    input_layer = Input(input)
    X= Conv3D(filters = 32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform') (input_layer)
    X=MaxPool3D(pool_size=(2, 2, 2))(X)
    X=Dropout(0.5)(X)
    X=Conv3D(filters = 64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(X)
    X=MaxPool3D(pool_size=(2, 2, 2))(X)
    X=Dropout(0.5)(X)
    X=Flatten()(X)
    X=Dense(256, activation='relu', kernel_initializer='he_uniform')(X)
    output_layer=Dense(units = classQTY, activation='softmax')(X)
    model = Model (inputs= input_layer, outputs = output_layer, name='3DCNN_v1')

    model.compile(optimizer=Adam(lr=rate),
                  loss='categorical_crossentropy', 
                   metrics=['accuracy'])
    return model

def model_v2 (input = (16,16,16,3), classQTY =10, rate = 0.05  ):

    input_layer = Input(input)

    X = Conv3D(filters=8, kernel_size=(3,3,3), activation='relu')(input_layer)
    X = Conv3D(filters=16, kernel_size=(3,3,3), activation='relu')(X)
    X = MaxPool3D(pool_size=(2,2,2))(X)

    X = Conv3D(filters=24, kernel_size=(3,3,3), activation='relu')(X)
    X = Conv3D(filters=32, kernel_size=(3,3,3), activation='relu')(X)
    X = MaxPool3D(pool_size=(2,2,2))(X)
    X = Flatten()(X)

    X = Dense(units=2048, activation='relu')(X)
    X = Dropout(0.4)(X)
    X= Dense(units=512, activation='relu')(X)
    X = Dropout(0.4)(X)
    output_layer = Dense(units= classQTY, activation='softmax')(X)

    model = Model(inputs = input_layer, outputs = output_layer, name='3DCNN_v2')
    #opt = SGD(lr=0.005, momentum=0.9)
    #model.compile( optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile( optimizer=Adadelta(lr=0.05), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])   
    return model

def model_v3(input = (16,16,16,3), classQTY =10, rate = 0.025 ):

    input_layer = Input(input)

    X = Conv3D(filters=16, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(input_layer)
    X = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform")(X)
    X = Dropout(0.1)(X)
    X = Conv3D(filters=32, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(X)
    X = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(X)
    X = Dropout(0.2)(X)

    X = MaxPool3D(pool_size=2,strides=2,padding='same')(X)
    
    X = Conv3D(filters=64, kernel_size=(5, 5, 5),strides=1,padding ='same',activation='relu')(X)
    X = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(X)
    X = Dropout(0.2)(X)

    X = Conv3D(filters=32, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(X)
    X = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(X)
    X = Dropout(0.2)(X)
    X = Flatten()(X)

    X = Dense(units=4096, activation='relu')(X)
    X = Dropout(0.05)(X)
    X = Dense(units=1024, activation='relu')(X)
    X = Dropout(0.05)(X)
    X = Dense(units=256, activation='relu')(X)  
    X = Dropout(0.05)(X)
    output_layer = Dense(units=classQTY, activation='softmax')(X)

    model = Model(inputs=input_layer, outputs=output_layer,name='3DCNN_v3')
    #opt = SGD(lr=rate, momentum=0.9)
    #model.compile( optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile( optimizer=Adadelta(lr=rate),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])  
    return model