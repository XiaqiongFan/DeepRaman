# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 15:19:56 2021

@author: xiaqiong
"""

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from sklearn import preprocessing
from sklearn.linear_model import enet_path
from scipy.sparse import spdiags,eye,csc_matrix, diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt 
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
import copy
import csv


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


def WhittakerSmooth(x, lamb, w):
    m=w.shape[0]
    W=spdiags(w,0,m,m)
    D=eye(m-1,m,1)-eye(m-1,m)
    return spsolve((W+lamb*D.transpose()*D),w*x)
    
def airPLS(x, lamb=10, itermax=10):
    m=x.shape[0]
    w=np.ones(m)
    for i in range(itermax):
        z=WhittakerSmooth(x,lamb,w)
        d=x-z
        if sum(abs(d[d<0]))<0.001*sum(abs(x)):
            break;
        w[d<0]=np.exp(i*d[d<0]/sum(d[d<0]))
        w[d>=0]=0
    return z

def airPLS_MAT(X, lamb=10, itermax=10):
    B=X.copy()
    for i in range(X.shape[0]):
        B[i,]=airPLS(X[i,],lamb,itermax)
    return X-B

def WhittakerSmooth_MAT(X, lamb=1):
    C=X.copy()
    w=np.ones(X.shape[1])
    for i in range(X.shape[0]):
        C[i,]=WhittakerSmooth(X[i,:], lamb, w)
    plt.figure()
    plt.plot(C.T)
    return C

if __name__ == '__main__':
 
    
    _custom_objects = {
    "SpatialPyramidPooling" :  SpatialPyramidPooling,
    }   
    
    
    datafile0 = u'./data/database.npy'
    spectrum_pure = np.load(datafile0) 

    datafile1 =u'./data/unknown.npy'
    spectrum_mix = np.load(datafile1)     

    csv_reader = csv.reader(open(u'./data/com.csv', encoding='utf-8'))
    DBcoms = [row for row in csv_reader]    

    spectrum_pure_sc =  copy.deepcopy(spectrum_pure)
    spectrum_mix_sc = copy.deepcopy(spectrum_mix)
    for i in range(spectrum_mix.shape[0]):
        spectrum_mix_sc[i,:] = spectrum_mix[i,:]/np.max(spectrum_mix[i,:])
    for i in range(spectrum_pure.shape[0]):
        spectrum_pure_sc[i,:] = spectrum_pure[i,:]/np.max(spectrum_pure[i,:])


    X = np.zeros((spectrum_mix_sc.shape[0]*spectrum_pure_sc.shape[0],2,881,1))

    for p in range(spectrum_mix_sc.shape[0]):
        for q in range(spectrum_pure_sc.shape[0]):
            X[int(p*spectrum_pure_sc.shape[0]+q),0,:,0] = spectrum_mix_sc[p,:]
            X[int(p*spectrum_pure_sc.shape[0]+q),1,:,0] = spectrum_pure_sc[q,:]
            

    re_model = tf.keras.models.load_model('./model/model.h5', custom_objects=_custom_objects)
    y = re_model.predict(X)

    spectrum_pure = WhittakerSmooth_MAT(spectrum_pure, lamb=1)
    spectrum_pure = airPLS_MAT(spectrum_pure, lamb=10, itermax=10)
    spectrum_mix = WhittakerSmooth_MAT(spectrum_mix, lamb=1)
    spectrum_mix = airPLS_MAT(spectrum_mix, lamb=10, itermax=10)
    

    for cc in range(spectrum_mix.shape[0]):
        com=[]
        coms = []
        ra2 = []
        for ss in range(cc*spectrum_pure.shape[0],(cc+1)*spectrum_pure.shape[0]):
            
            if y[ss,1]>=0.5:
                com.append(ss%spectrum_pure.shape[0])


        X = spectrum_pure[com]
        coms = [DBcoms[com[h]] for h in range(len(com))]

        _, coefs_lasso, _ = enet_path(X.T, spectrum_mix[cc,:], l1_ratio=0.96,
                                  positive=True, fit_intercept=False)
        ratio = coefs_lasso[:, -1]
        ratio_sc = copy.deepcopy(ratio)
                
        for ss2 in range(ratio.shape[0]):
            ratio_sc[ss2]=ratio[ss2]/np.sum(ratio)

            
        print('The',cc, 'spectra may contain:',coms)
        print('The corresponding ratio is:', ratio_sc)

        



