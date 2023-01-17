# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 19:29:07 2021

@author: admin
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import tensorflow.keras.backend as K
import os
import time
from tensorflow.keras.layers import Layer

#   Define SpatialPyramidPooling
class SpatialPyramidPooling(Layer):

    def __init__(self, pool_list, **kwargs):

        self.dim_ordering = K.image_data_format()
        assert self.dim_ordering in {'channels_last', 'channels_first'}, 'dim_ordering must be in {tf, th}'

        self.pool_list = pool_list

        self.num_outputs_per_channel = sum([i * i for i in pool_list])

        super(SpatialPyramidPooling, self).__init__(**kwargs)

    def build(self, input_shape):
            self.nb_channels = input_shape[3]


    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_outputs_per_channel)

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):

        input_shape = K.shape(x)


        num_rows = input_shape[1]
        num_cols = input_shape[2]

        row_length = [K.cast(num_rows, 'float32') / i for i in self.pool_list]
        col_length = [K.cast(num_cols, 'float32') / i for i in self.pool_list]

        outputs = []

      
        for pool_num, num_pool_regions in enumerate(self.pool_list):

            for ix in range(num_pool_regions):
                for iy in range(num_pool_regions):
                    x1 = ix * col_length[pool_num]
                    x2 = ix * col_length[pool_num] + col_length[pool_num]
                    y1 = iy * row_length[pool_num]
                    y2 = iy * row_length[pool_num] + row_length[pool_num]

                    x1 = K.cast(K.round(x1), 'int32')
                    x2 = K.cast(K.round(x2), 'int32')
                    y1 = K.cast(K.round(y1), 'int32')
                    y2 = K.cast(K.round(y2), 'int32')
    
                    new_shape = [input_shape[0], y2 - y1,
                                 x2 - x1, input_shape[3]]
                    x_crop = x[:, y1:y2, x1:x2, :]
                    xm = K.reshape(x_crop, new_shape)
                    pooled_val = K.max(xm, axis=(1, 2))
                    outputs.append(pooled_val)

        if self.dim_ordering == 'channels_first':
            outputs = K.concatenate(outputs)
        elif self.dim_ordering == 'channels_last':
            outputs = K.concatenate(outputs,axis = 0)
            outputs = K.reshape(outputs,(self.num_outputs_per_channel,input_shape[0], self.nb_channels))
            outputs = K.permute_dimensions(outputs,(1,0,2))
            outputs = K.reshape(outputs,(input_shape[0], self.num_outputs_per_channel * self.nb_channels))
        return outputs


def SSPmodel(X):
    # X：processed training data [:,2,N,1],N represents the length of spectra
    # inputA: spectra from database
    # inputB：spectra of unknown
    

    inputs = tf.keras.layers.Input((2,None,1))
    
    inputA = inputs[:,0,:]
    inputB = inputs[:,1,:]

    
    convA1 = tf.keras.layers.Conv1D(32, kernel_size=(7), strides=(1),padding = 'same', kernel_initializer = 'he_normal')(inputA)    
    convA1 = tf.keras.layers.BatchNormalization()(convA1)
    convA1 = tf.keras.layers.Activation('relu')(convA1) 
    poolA1 = tf.keras.layers.MaxPooling1D(3)(convA1)


    convB1 = tf.keras.layers.Conv1D(32, kernel_size=(7), strides=(1),padding = 'same', kernel_initializer = 'he_normal')(inputB)    
    convB1 = tf.keras.layers.BatchNormalization()(convB1)
    convB1 = tf.keras.layers.Activation('relu')(convB1) 
    poolB1 = tf.keras.layers.MaxPooling1D(3)(convB1)

    con = tf.keras.layers.concatenate([poolA1,poolB1],2)
    con = tf.expand_dims(con, -1)
    
    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(7,7), strides=(2,2),padding = 'same', kernel_initializer = 'he_normal')(con)    
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation('relu')(conv1) 
    
    spp = SpatialPyramidPooling([1, 2, 3, 4])(conv1)

    full1 = tf.keras.layers.Dense(1024, activation='relu')(spp)
    drop1 = tf.keras.layers.Dropout(0.5)(full1)


    outputs = tf.keras.layers.Dense(2, activation='sigmoid')(drop1)#sigmoid
    model = tf.keras.models.Model(inputs,outputs)
    return model


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False
                
def cleardir(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def reader(dataX,dataY,batch_size):
    step = 0
    steps = dataX.shape[0]//batch_size    
    while True:

        dataX_batch = dataX[int(step*batch_size):int((step+1)*batch_size)]
        dataY_batch = dataY[int(step*batch_size):int((step+1)*batch_size)]
       
        yield dataX_batch,dataY_batch
        step = (step+1)% steps
    
def load_data(datapath):
    
    datafileX = datapath+'/training_X.npy'
    Xtrain0 = np.load(datafileX)   
    datafileY = datapath+'/training_Y.npy'
    Ytrain0 = np.load(datafileY)

    XtrainA,YtrainA=process(Xtrain0,Ytrain0)

    XXtrain=XtrainA[0:int(0.9*XtrainA.shape[0])]
    YYtrain=YtrainA[0:int(0.9*XtrainA.shape[0])]

    XXvalid=XtrainA[int(0.9*XtrainA.shape[0]):]
    YYvalid=YtrainA[int(0.9*XtrainA.shape[0]):]    


    return XXtrain,YYtrain,XXvalid,YYvalid

def process(Xtrain,Ytrain):
    for i in range(Xtrain.shape[0]):
        for j in range(Xtrain.shape[1]):
            for k in range(Xtrain.shape[2]):
                Xtrain[i,j,k,:]=Xtrain[i,j,k,:]/np.max(Xtrain[i,j,k,:])
    Xtrain = Xtrain.reshape(Xtrain.shape[0]*Xtrain.shape[1],Xtrain.shape[2],Xtrain.shape[3],1)
    Ytrain = Ytrain.reshape(Ytrain.shape[0]*Ytrain.shape[1],1)
    
    Ytrain = tf.keras.utils.to_categorical(Ytrain)

    Xtrain,Ytrain = randomize(Xtrain,Ytrain)

    return Xtrain,Ytrain




if __name__ == '__main__':


    _custom_objects = {
    "SpatialPyramidPooling" :  SpatialPyramidPooling,
    }     

    # define the data path and save path
    datapath = './data'

    start = time.time()
    savepath = './model'
    mkdir(savepath)

    # load data
    Xtrain,Ytrain,Xvalid,Yvalid = load_data(datapath)  
 
    tf.keras.backend.clear_session()
    ops.reset_default_graph()
    
    # model training
    model = SSPmodel(Xtrain)
    model.compile(optimizer=tf.keras.optimizers.SGD(),
                 loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

            
    batch_size = 200
    epochs = 50
    history = model.fit_generator(generator = reader(Xtrain,Ytrain,batch_size),
                        epochs = epochs,steps_per_epoch=Xtrain.shape[0]//batch_size,
                        validation_data = reader(Xvalid,Yvalid,200),
                        validation_steps = 1)
       
    # learning cures        
    fig = plt.figure(figsize = (6,4.5))
    ax = fig.add_subplot(111)       
    ax.set_ylabel('Accuracy',size=15)
    ax.set_xlabel('Epoch',size=15)       
    lns1 = plt.plot(history.history['accuracy'],label = 'Acc_training',color='r')
    lns2 = plt.plot(history.history['val_accuracy'],label = 'Acc_validation',color='g')
    ax2 = ax.twinx()  
    ax2.set_ylabel('Loss',size=15)
    lns3 = plt.plot(history.history['loss'],label = 'Loss_training',color='b')
    lns4 = plt.plot(history.history['val_loss'],label = 'Loss_validation',color='orange')
    lns = lns1+lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=7)
    plt.show()
  
    model.save(savepath+'/model.h5')

    del model
    end = time.time()
    print('Training finished, time:%.2fSeconds'%(end-start))

