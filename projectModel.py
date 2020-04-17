import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta, Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model

def simpleModel():

    input_layer = Input((16,16,16,3))

    #convolution layer
    conv_layer_one = Conv3D(filters=8, kernel_size=(3,3,3), activation='relu')(input_layer)
    conv_layer_two = Conv3D(filters=16, kernel_size=(3,3,3), activation='relu')(conv_layer_one)

    #pooling layer
    pooling_layer_one = MaxPool3D(pool_size=(2,2,2))(conv_layer_two)

    #convolution layer
    conv_layer_three = Conv3D(filters=24, kernel_size=(3,3,3), activation='relu')(pooling_layer_one)
    conv_layer_four = Conv3D(filters=32, kernel_size=(3,3,3), activation='relu')(conv_layer_three)

    #pooling layer
    pooling_layer_two = MaxPool3D(pool_size=(2,2,2))(conv_layer_four)
    flatten_layer = Flatten()(pooling_layer_two)

    #Fully Connected layers
    dense_layer_one = Dense(units=2048, activation='relu')(flatten_layer)
    dense_layer_one = Dropout(0.4)(dense_layer_one)
    dense_layer_two = Dense(units=512, activation='relu')(dense_layer_one)
    dense_layer_two = Dropout(0.4)(dense_layer_two)
    output_layer = Dense(units=10, activation='softmax')(dense_layer_two)

    model = Model(inputs = input_layer, outputs = output_layer)
    #opt = SGD(lr=0.005, momentum=0.9)
    #model.compile( optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile( optimizer=Adadelta(lr=0.05), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])   
    return model

def model_v2():

    model = Sequential()
    input_layer = Input((16,16,16,3))
    X= Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform') (input_layer)
    X=MaxPool3D(pool_size=(2, 2, 2))(X)
    X=Dropout(0.5)(X)
    X=Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform')(X)
    X=MaxPool3D(pool_size=(2, 2, 2))(X)
    X=Dropout(0.5)(X)
    X=Flatten()(X)
    X=Dense(256, activation='relu', kernel_initializer='he_uniform')(X)
    output_layer=Dense(units = 10, activation='softmax')(X)
    model = Model (inputs= input_layer, outputs = output_layer)

    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy', 
                   metrics=['accuracy'])
    return model

def model_v3():
    ## input layer
    input_layer = Input((16, 16, 16, 3))

    ## convolutional layers
    conv_layer1 = Conv3D(filters=16, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(input_layer)
    conv_layer1 = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform")(conv_layer1)
    conv_layer1 = Dropout(0.1)(conv_layer1)
    conv_layer2 = Conv3D(filters=32, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(conv_layer1)
    conv_layer2 = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(conv_layer2)
    conv_layer2 = Dropout(0.2)(conv_layer2)
    ## add max pooling to obtain the most imformatic features
    pooling_layer1 = MaxPool3D(pool_size=2,strides=2,padding='same')(conv_layer2)
    
    conv_layer3 = Conv3D(filters=64, kernel_size=(5, 5, 5),strides=1,padding ='same',activation='relu')(pooling_layer1)
    conv_layer3 = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(conv_layer3)
    conv_layer3 = Dropout(0.2)(conv_layer3)
    conv_layer4 = Conv3D(filters=32, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(conv_layer3)
    conv_layer4 = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(conv_layer4)
    conv_layer4 = Dropout(0.2)(conv_layer4)
    ##pooling_layer3 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)
    ## perform batch normalization on the convolution outputs before feeding it to MLP architecture
    flatten_layer = Flatten()(conv_layer4)

    ## create an MLP architecture with dense layers : 4096 -> 512 -> 10
    ## add dropouts to avoid overfitting / perform regularization
    dense_layer1 = Dense(units=4096, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.05)(dense_layer1)
    dense_layer2 = Dense(units=1024, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.05)(dense_layer2)
    dense_layer3 = Dense(units=256, activation='relu')(dense_layer2)
    dense_layer3 = Dropout(0.05)(dense_layer3)
    output_layer = Dense(units=10, activation='softmax')(dense_layer3)

    ## define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer,name='3DCNN')
    #adam = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, amsgrad=True)
    #model.compile(loss=categorical_crossentropy, optimizer=adam, metrics=['acc'])
    model.compile( optimizer=Adadelta(lr=0.05),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])  
    return model


def model_v4():
    ## input layer
    input_layer = Input((18, 18,18, 3))

    ## convolutional layers
    conv_layer1 = Conv3D(filters=16, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(input_layer)
    conv_layer1 = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform")(conv_layer1)
    conv_layer1 = Dropout(0.1)(conv_layer1)
    conv_layer2 = Conv3D(filters=32, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(conv_layer1)
    conv_layer2 = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(conv_layer2)
    conv_layer2 = Dropout(0.2)(conv_layer2)
    ## add max pooling to obtain the most imformatic features
    pooling_layer1 = MaxPool3D(pool_size=2,strides=2,padding='same')(conv_layer2)
    
    conv_layer3 = Conv3D(filters=64, kernel_size=(5, 5, 5),strides=1,padding ='same',activation='relu')(pooling_layer1)
    conv_layer3 = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(conv_layer3)
    conv_layer3 = Dropout(0.2)(conv_layer3)
    conv_layer4 = Conv3D(filters=32, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(conv_layer3)
    conv_layer4 = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(conv_layer4)
    conv_layer4 = Dropout(0.2)(conv_layer4)
    ##pooling_layer3 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)
    ## perform batch normalization on the convolution outputs before feeding it to MLP architecture
    flatten_layer = Flatten()(conv_layer4)

    ## create an MLP architecture with dense layers : 4096 -> 512 -> 10
    ## add dropouts to avoid overfitting / perform regularization
    dense_layer1 = Dense(units=4096, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.05)(dense_layer1)
    dense_layer2 = Dense(units=1024, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.05)(dense_layer2)
    dense_layer3 = Dense(units=256, activation='relu')(dense_layer2)
    dense_layer3 = Dropout(0.05)(dense_layer3)
    output_layer = Dense(units=4, activation='softmax')(dense_layer3)

    ## define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer,name='3DCNN')
    #adam = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, amsgrad=True)
    #model.compile(loss=categorical_crossentropy, optimizer=adam, metrics=['acc'])
    model.compile( optimizer=Adadelta(lr=0.025),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])  
    return model

def model_v5(input = (16,16,16,3), classQTY =4 ):
    ## input layer
    input_layer = Input(input)

    ## convolutional layers
    conv_layer1 = Conv3D(filters=16, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(input_layer)
    conv_layer1 = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform")(conv_layer1)
    conv_layer1 = Dropout(0.1)(conv_layer1)
    conv_layer2 = Conv3D(filters=32, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(conv_layer1)
    conv_layer2 = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(conv_layer2)
    conv_layer2 = Dropout(0.2)(conv_layer2)
    ## add max pooling to obtain the most imformatic features
    pooling_layer1 = MaxPool3D(pool_size=2,strides=2,padding='same')(conv_layer2)
    
    conv_layer3 = Conv3D(filters=64, kernel_size=(5, 5, 5),strides=1,padding ='same',activation='relu')(pooling_layer1)
    conv_layer3 = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(conv_layer3)
    conv_layer3 = Dropout(0.2)(conv_layer3)
    conv_layer4 = Conv3D(filters=32, kernel_size=(3, 3, 3),strides=1,padding ='same',activation='relu')(conv_layer3)
    conv_layer4 = BatchNormalization(momentum=0.15, epsilon=1e-5, gamma_initializer="uniform")(conv_layer4)
    conv_layer4 = Dropout(0.2)(conv_layer4)
    ##pooling_layer3 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer4)
    ## perform batch normalization on the convolution outputs before feeding it to MLP architecture
    flatten_layer = Flatten()(conv_layer4)

    ## create an MLP architecture with dense layers : 4096 -> 512 -> 10
    ## add dropouts to avoid overfitting / perform regularization
    dense_layer1 = Dense(units=4096, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.05)(dense_layer1)
    dense_layer2 = Dense(units=1024, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.05)(dense_layer2)
    dense_layer3 = Dense(units=256, activation='relu')(dense_layer2)
    dense_layer3 = Dropout(0.05)(dense_layer3)
    output_layer = Dense(units=classQTY, activation='softmax')(dense_layer3)

    ## define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer,name='3DCNN')
    #adam = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, amsgrad=True)
    #model.compile(loss=categorical_crossentropy, optimizer=adam, metrics=['acc'])
    model.compile( optimizer=Adadelta(lr=0.025),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])  
    return model