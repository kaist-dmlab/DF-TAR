###############################
### IMPORT REQUIRED MODULES ###
###############################
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, GRU, TimeDistributed, RepeatVector, BatchNormalization, Conv1D, MaxPooling1D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Reshape, Flatten, Input, GaussianNoise, Attention, Bidirectional, Embedding, Dot, GlobalMaxPooling2D
from tensorflow.keras.layers import Concatenate, Multiply, Add, Activation, ReLU, ConvLSTM2D, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from Transformer import TransformerBlock


###########################
### DEFINE DF-TAR MODEL ###
###########################
def DF_TAR(n_steps, length, n_districts, n_features, conv_hidden=256, gru_hidden=256, fc_hidden=1024, batchnorm=True):
    # Input Layers
    risk = Input(shape=[length, n_districts, n_features['risk']])
    E_static = Input(shape=[length, n_districts, n_features['E_static']])
    E_dynamic = Input(shape=[length, n_districts, n_features['E_dynamic']])
    c = Input(shape=[length, n_districts, n_features['c']])

    # Convolutional Block for Static Environmental Features
    h_static = Conv2D(conv_hidden, 9, padding='same', activation='relu')(E_static)
    h_static = MaxPooling2D(3, padding='same')(h_static)
    h_static = Conv2D(conv_hidden, 4, padding='same', activation='relu')(h_static)
    h_static = MaxPooling2D(3, padding='same')(h_static)
    h_static = Conv2D(conv_hidden, 3, padding='same', activation='relu')(h_static)
    h_static = MaxPooling2D(3, padding='same')(h_static)
    h_static = Conv2D(conv_hidden, 3, padding='same', activation='relu')(h_static)
    h_static = GlobalMaxPooling2D()(h_static)
    h_static = Flatten()(h_static)
    h_static = Dense(conv_hidden)(h_static)
    h_static = RepeatVector(length)(h_static)

    # Recurrent Block for Historical Risk Scores
    h_risk = Reshape([length, n_districts * n_features['risk']])(risk)
    h_risk = TransformerBlock(n_districts * n_features['risk'], n_features['risk'], 256)(h_risk)
    h_risk = Concatenate()([h_risk, h_static])
    h_risk = GRU(gru_hidden)(h_risk)

    # Recurrent Block for Dynamic Environmental Features
    h_dynamic = Reshape([length, n_districts * n_features['E_dynamic']])(E_dynamic)
    h_dynamic = TransformerBlock(n_districts * n_features['E_dynamic'], n_features['E_dynamic'], 256)(h_dynamic)
    h_dynamic = Concatenate()([h_dynamic, h_static])
    h_dynamic = GRU(gru_hidden)(h_dynamic)

    # Recurrent Block for Dangerous Driving Statistics
    h_case = Reshape([length, n_districts * n_features['c']])(c)
    h_case = TransformerBlock(n_districts * n_features['c'], n_features['c'], 256)(h_case)
    h_case = Concatenate()([h_case, h_static])
    h_case = GRU(gru_hidden)(h_case)

    # Fusion Block
    h_concat = Concatenate()([h_dynamic, h_risk, h_case])
    tanh = Activation('tanh')(h_concat)
    sigmoid = Activation('sigmoid')(h_concat)
    h_fusion = Multiply()([tanh, sigmoid])

    # Fully-Connected Block for Final Prediction
    f = Dense(fc_hidden)(h_fusion)
    if batchnorm:
        f = BatchNormalization()(f)
    f = Activation('relu')(f)
    
    f = Dense(fc_hidden)(f)
    if batchnorm:
        f = BatchNormalization()(f)
    f = Activation('relu')(f)
    
    f = Dense(fc_hidden)(f)
    if batchnorm:
        f = BatchNormalization()(f)
    f = Activation('relu')(f)
    

    # Predicted Risk Scores optimized by MAE
    y_mae = Dense(n_steps * n_districts)(f)
    y_mae = Reshape([n_steps, n_districts], name='Y_MAE')(y_mae)
    
    # Predicted Risk Scores optimized by MSE
    y_mse = Dense(n_steps * n_districts)(f)
    y_mse = Reshape([n_steps, n_districts], name='Y_MSE')(y_mse)
    
    return Model(inputs=[risk, E_static, E_dynamic, c], outputs=[y_mae, y_mse])