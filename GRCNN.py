#!/usr/bin/env python
# coding: utf-8


import os
import csv
import time
import random
import cPickle
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pylab
import matplotlib.pyplot as plt
#from IPython.display import Image as IPImage
from sklearn import preprocessing
#normalized_data, norm = sklearn.preprocessing.normalize(data, norm='l2', axis=0, copy=False, return_norm=True) #version update needed

import numpy as np
import tensorflow as tf
#from tensorflow.models.rnn import rnn, rnn_cell

import utils
import models



def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())



def test(model, save_path, path, model_name, epoch, data_columns, target_features, target_delay, test_data, time_point, time_step, 
         scaler, xrange=5000, file_name='test_output', load_model=False):
    if load_model :
        model.load(os.path.join(save_path, path, model_name, str(epoch+1)+'.ckpt'))
    test_file = os.path.join(save_path, path, model_name, str(epoch+1)+'epoch-'+file_name+'.npz')
    if os.path.exists(test_file):
        test_output = np.load(test_file)['arr_0']
        test_output = test_output.reshape([-1,len(target_delay)])
        print 'test output {} loaded.'.format(test_output.shape)
    else:
        test_output = []
        for batch in utils.iterate_3d_2(inputs=np.delete(test_data, [data.columns.tolist().index(i) for i in unused_features], axis=2),
                                targets=test_data[:,:,[data.columns.tolist().index(i) for i in target_features]],
                                target_delay=np.array(target_delay), batch_size=batch_size, length = max(test_data.shape), 
                                time_point=time_point, time_step=time_step, time_int=1):
            test_in, test_target = batch
            test_output.append(model.reconstruct(test_in))
        test_output = np.asarray(test_output)
        print test_output.shape
        test_output = test_output.reshape([-1,(len(target_features)*len(target_delay))])
        print test_output.shape
        np.savez(os.path.join(save_path,path,model_name, str(epoch+1)+'epoch-'+file_name+'.npz'), [test_output])
        print 'test output saved.'
    for i in  range(len(target_delay)):
        #xrange=10000
        if xrange is None:
            xrange = np.max(test_output.shape)
        target_data = test_data[0,
                                (time_point*time_step-1 + target_delay[i]):(time_point*time_step-1 + target_delay[i]+test_output.shape[0]),
                                np.where(data.columns==target_features[0])[0][0]]
        cp = np.append([False],np.diff(target_data)!=0)
        cp_rmse = rmse(target_data[cp]*scaler.scale_[np.where(data_columns==target_features[0])[0][0]],
                       test_output[cp,i]*scaler.scale_[np.where(data_columns==target_features[0])[0][0]])
        all_rmse = rmse(test_output[:, i]*scaler.scale_[np.where(data_columns==target_features[0])[0][0]], 
                        test_data[0,
                                  (time_point*time_step-1 + target_delay[i]):\
                                  (time_point*time_step-1 + target_delay[i]+test_output.shape[0]), 
                                  np.where(data_columns==target_features[0])[0][0]] * scaler.scale_[np.where(data.columns==target_features[0])[0][0]] )
        test_output_roll = pd.rolling_mean(test_output, batch_size)
        #
        print test_output.shape
        plt.figure(figsize=(15,6)) 
        plt.plot(range(time_point*time_step-1 + target_delay[i], time_point*time_step-1 + target_delay[i]+xrange),
                 test_output[0:xrange,i]*\
                             scaler.scale_[np.where(data_columns==target_features[0])[0][0]]+\
                             scaler.mean_[np.where(data_columns==target_features[0])[0][0]], 
                 label=target_features[0]+" " +str(target_delay[i])+'min pred', color='red')
        plt.plot(test_data[0, 0:xrange,
                           np.where(data_columns==target_features[0])[0][0]]*\
                             scaler.scale_[np.where(data_columns==target_features[0])[0][0]]+\
                             scaler.mean_[np.where(data_columns==target_features[0])[0][0]],
         label=target_features[0], color='blue')
        plt.legend()#fontsize=11)
        #plt.ylim([1440, 1560])
        plt.xlim([0, time_point*time_step-1 + target_delay[i]+xrange])
        plt.figtext(0.1, 0.01, 'all_rmse: ' +str(all_rmse))
        plt.figtext(0.7, 0.01, 'cp_rmse: ' +str(cp_rmse))
        #plt.savefig(os.path.join(save_path, path, model_name, str(xrange)+'-'+str(epoch)+'epoch-'+str(target_delay[i])+'m_test_output.png'))
        plt.savefig(os.path.join('./', path, model_name, str(xrange)+'-'+str(epoch+1)+'epoch-'+str(target_delay[i])+'m_'+file_name+'.png'))
        plt.close()



class GRCNN(object):
    def __init__(self, batch_size=128, time_point=1024, in_channels = 126, out_channels=256, ch_multiplier=None, 
                 cluster=None, rrcl_iter=[2,2,2,2], rrcl_num=4, forward_layers=[200,3], pool=['n','p','p','p','c'],
                 use_batchnorm=True, scale=1, offset=0.01, epsilon=0.01, nonlinearity=None, keep_probs=None, 
                 std=0.01, w_filter_size=9, p_filter_size=4, l_rate=0.01, l_decay=0.95, l_step=1000, 
                 optimizer='RMSProp', opt_epsilon=0.1, decay=0.9, momentum=0.9, tpa_coeff=0.0001):
        print 'start initializing...'
        self.batch_size = batch_size
        self.time_point = time_point
        self.in_channels = in_channels
        #self.out_channels= ch_multiplier
        self.out_channels = out_channels
        if ch_multiplier!=None:
            print'\'ch_multiplier\' is depreciated. Use \'out_channels\''
            self.out_channels = ch_multiplier
        self.cluster = cluster
        self.rrcl_iter = rrcl_iter
        self.rrcl_num = rrcl_num
        self.use_batchnorm = use_batchnorm
        self.offset = offset
        self.scale = scale
        self.epsilon = epsilon
        self.nonlinearity = nonlinearity
        #self.keep_probs = keep_probs
        self.use_dropout = not (keep_probs == None or keep_probs == [1.0 for i in range(len(keep_probs))])
        #if keep_probs == None:
        #    self.keep_probs = [1.0 for i in range(1+rrcl_num+len(forward_layers)-1)]
        if self.use_dropout and len(keep_probs) != (1 + rrcl_num + len(forward_layers)-1):
            raise ValueError('\'keep_probs\' length is wrong')
        self.std = std
        self.w_filter_size = w_filter_size
        self.p_filter_size = p_filter_size
        t=0
        for i in range(len(np.unique(self.cluster)) ):
            t = t+ self.out_channels*np.sum(self.cluster==i)/self.in_channels
        self.ch_sum = t
        self.forward_layers = [t] + forward_layers ################
        self.pool = pool
        if len(self.pool) != rrcl_num+1:
            raise ValueError('Parameter \'pool\' length does not match with the model shape.')
        global_step = tf.Variable(0, trainable=False)
        self.l_rate = tf.train.exponential_decay(l_rate, global_step, l_step, l_decay, staircase=True)
        self.decay = decay
        self.momentum = momentum
        
        self.y = tf.placeholder(tf.float32, [None, self.forward_layers[-1]], name='y');
        self.x = [tf.placeholder(tf.float32, [None, 1, time_point, np.sum(cluster==i)], name='x'+str(i)) for i 
                  in range(len(np.unique(cluster))) ]
        self.keep_probs = tf.placeholder(tf.float32, [1+rrcl_num+len(forward_layers)-1], name='keep_probs')
        self.keep_probs_values = keep_probs
        print '  start building...'
        self.build_model( )
        print '  done.'
        # Define loss and optimizer, minimize the squared error
        #self.cost = tf.reduce_mean(tf.pow(self.y - self.output, 2))
        #self.cost = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.output), reduction_indices=[1]))
        self.cost = tf.reduce_mean(tf.pow(self.y - self.output_layer, 2))
        if optimizer=='Adam':
            self.optimizer = tf.train.AdamOptimizer(self.l_rate, epsilon=opt_epsilon).minimize(self.cost, global_step=global_step)
        else :#optimizer=='RMSProp':
            self.optimizer = tf.train.RMSPropOptimizer(self.l_rate, 
                                                   decay=self.decay, 
                                                   momentum=self.momentum).minimize(self.cost, global_step = global_step)
        # Initializing the tensor flow variables
        #init = tf.initialize_all_variables()
        
        # Launch the session
        self.session_conf = tf.ConfigProto()
        self.session_conf.gpu_options.allow_growth = True

        self.sess = tf.InteractiveSession(config=self.session_conf)
        #self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10000)
        print'done.'
        
    def build_model(self):
        #self.weights, self.biases = self.init_weights()
        length = self.time_point ##length
        filter_size = self.w_filter_size
        while filter_size> length:
            filter_size = filter_size/2
        self.conv1=[]
        networks=[]
        for i in range( len(np.unique(self.cluster))):
            """
            conv2d(input, filter, strides=[1,1,1,1], padding='SAME', nonlinearity=None, use_dropout=True, keep_prob=1.0, 
                   use_batchnorm=True, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='conv2d_default'):
            """
            #print i
            #print self.x[i],
            #print [1, self.w_filter_size, np.sum(self.cluster==i), self.out_channels*np.sum(self.cluster==i)/self.in_channels]
            conv1 = models.conv2d(self.x[i], 
                           weight_size=[1, filter_size, np.sum(self.cluster==i), self.out_channels*np.sum(self.cluster==i)/self.in_channels], 
                           nonlinearity=self.nonlinearity, 
                           pool = self.pool[0],
                           pool_size = self.p_filter_size,
                           use_dropout=self.use_dropout,
                           keep_prob=self.keep_probs[0], 
                           use_batchnorm=self.use_batchnorm, 
                           std=self.std, 
                           offset=self.offset, 
                           scale=self.scale, 
                           epsilon=self.epsilon, 
                           name='conv2d_cluster'+str(i))
            self.conv1.append(conv1)
            networks.append(conv1)
            #print conv1.get_layer()
        #(batch, time, in_ch, ch_mult)
        print '    conv done.    {}'.format(conv1.get_layer())
        """
        self.conv1p = tf.nn.max_pool(value=self.conv1, 
                                    ksize=[1,1,4,1], 
                                    strides=[1,1,4,1], 
                                    padding='SAME')
        """
        #output: (batch_size, 1, in_width, out_channels*in_channels)
        
        """
        RCL(input, filter, strides=[1,1,1,1], padding='SAME', num_iter=3, nonlinearity=None, use_dropout=True, keep_prob=1.0, 
            use_batchnorm=True, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='RCL_default'):
        """
        #networks = self.conv1.get_layer()
        self.rrcls = []
        for r in range(self.rrcl_num):
            rrcl = []
            while filter_size> length:
                filter_size = filter_size/2
            for i in range( len(np.unique(self.cluster))):
                #print ' cluster{} start'.format(i),
                tmp = models.RCL(input = networks[i].get_layer(), 
                          weight_size = [1, filter_size, self.out_channels*np.sum(self.cluster==i)/self.in_channels, 
                                         self.out_channels*np.sum(self.cluster==i)/self.in_channels], 
                          num_iter = self.rrcl_iter[r], 
                          nonlinearity = self.nonlinearity, 
                          use_dropout = self.use_dropout,
                          keep_prob = self.keep_probs[1+r], 
                          use_batchnorm = self.use_batchnorm,
                          std=self.std,
                          offset=self.offset,
                          scale=self.scale,
                          epsilon=self.epsilon, 
                          pool=self.pool[r+1],
                          pool_size=self.p_filter_size,
                          name='RCL'+str(r)+'_cluster'+str(i))
                rrcl.append(tmp)
                #print 'done'
            networks = rrcl
            self.rrcls.append(rrcl)
            length = length/self.p_filter_size
            print'    rrcl{} done'.format(r),
            print'    {}'.format(rrcl[-1].get_layer())
        #
        networks=[]
        for i in range(len(rrcl)):
            networks.append( rrcl[i].get_layer())
            #print networks[i]
        self.concat = tf.concat(3, networks)
        print '    concatenated to {}'.format(self.concat)
        
        network = tf.reshape(self.concat, shape=[-1, self.ch_sum])# * self.keep_probs[1]]) ###
        self.flatten = network
        print '    flatten to {}'.format(self.flatten)
        """
        (input, weight, nonlinearity=None, use_dropout=False, keep_prob=1.0, 
        use_batchnorm=False, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='feedforward_default')
        """
        if len(self.forward_layers) == 2:
            network = models.feedforward(input = network,
                                  weight_size=[self.forward_layers[0], self.forward_layers[1]],
                                  nonlinearity=None,
                                  use_dropout = False, 
                                  use_batchnorm = False,
                                  std=self.std,
                                  offset=self.offset,
                                  scale=self.scale,
                                  epsilon=self.epsilon, 
                                  name='output')
            self.output = network#.get_layer()
            self.output_layer = network.get_layer()
            print'    feedforward {} done, {}'.format(i+1, self.output_layer)
            print'    model built'
        else:
            self.forwards=[]
            for i in range(len(self.forward_layers)-1 -1):
                network  = models.feedforward(input = network, 
                                              weight_size=[self.forward_layers[i], self.forward_layers[i+1]],
                                              nonlinearity=self.nonlinearity, 
                                              use_dropout = self.use_dropout, 
                                              keep_prob = self.keep_probs[1+r], 
                                              use_batchnorm = self.use_batchnorm,
                                              std=self.std,
                                              offset=self.offset,
                                              scale=self.scale,
                                              epsilon=self.epsilon, 
                                              name='forward'+str(i))
                self.forwards.append(network)
                network = network.get_layer()
                print'    feedforward {} done, {}'.format(i, network)
            network =  models.feedforward(input = network,
                                         weight_size=[self.forward_layers[-2], self.forward_layers[-1]],
                                         nonlinearity=None,
                                         use_dropout = False, 
                                         use_batchnorm = False,
                                         std=self.std,
                                         offset=self.offset,
                                         scale=self.scale,
                                         epsilon=self.epsilon, 
                                         name='output')
            self.output = network#.get_layer()
            self.output_layer = network.get_layer()
            print'    feedforward {} done, {}'.format(i+1, self.output_layer)
            print'    model built'
            
    def train(self, data, target, keep_probs=None):
        ## data: [batch, time_idx]
        ## x: [batch, in_height, in_width, in_channels]
        train_feed_dict = {self.x[i]:data[:,:,:,np.where(self.cluster==i)[0]] for i in range(len(np.unique(self.cluster))) }
        train_feed_dict.update({self.y:target})
        if keep_probs is None:
            train_feed_dict.update({self.keep_probs:self.keep_probs_values})
        else:
            self.keep_probs_values = keep_probs
            train_feed_dict.update({self.keep_probs:keep_probs})
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                             feed_dict=train_feed_dict
                                            )
        return cost
    
    def test(self, data, target, keep_probs=None):
        test_feed_dict = {self.x[i]:data[:,:,:,np.where(self.cluster==i)[0]] for i in range(len(np.unique(self.cluster))) }
        test_feed_dict.update({self.y:target})
        if keep_probs is None:
            test_feed_dict.update({self.keep_probs:self.keep_probs_values})
        else:
            self.keep_probs_values = keep_probs
            test_feed_dict.update({self.keep_probs:keep_probs})
        cost = self.sess.run(self.cost, 
                             feed_dict=test_feed_dict
                            )
        return cost
    
    def reconstruct(self, data, keep_probs=None):
        recon_feed_dict = {self.x[i]:data[:,:,:,np.where(self.cluster==i)[0]] for i in range(len(np.unique(self.cluster))) }
        if keep_probs is None:
            recon_feed_dict.update({self.keep_probs:self.keep_probs_values})
        else:
            self.keep_probs_values = keep_probs
            recon_feed_dict.update({self.keep_probs:keep_probs})
        return self.sess.run(self.output_layer, 
                             feed_dict=recon_feed_dict
                            )
    
    def save(self, save_path='./model.ckpt'):
        saved_path = self.saver.save(self.sess, save_path)
        print("Model saved in file: %s"%saved_path)
        
    def load(self, load_path = './model.ckpt'):
        self.saver.restore(self.sess, load_path)
        print("Model restored")
    
    def terminate(self):
        self.sess.close()
        tf.reset_default_graph()




# Main

"""
Load Data
"""
# os.getcwd() #current path
#data_path = os.path.join('/data2/data','data3')
#data = pd.read_csv( os.path.join(data_path,'toy_data.csv'))
data_path = '/data1/data-v0/'
data = pd.read_csv(os.path.join(data_path,'data.csv'))
#dataColumns = data.columns.tolist()
print("data shape: "+data.shape)
# data: (time_index, features)

cluster =  pd.read_csv(os.path.join(data_path, 'data_SpectralClustering10', 'cluster.csv'))
print("cluster shape: "+cluster.shape)


if data.shape[-1] != cluster.shape[-1]:
    raise ValueError('wrong cluster file 1')
if sum(data.columns == cluster.columns) != data.shape[-1]:
    raise ValueError('wrong cluster file 2. Possibly not in order.')



# standardization
scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(data)
# scaler.mean_.shape, scaler.scale_.shape #mean, std

train_data = data[:int(data.shape[0]*0.8)]
valid_data = data[int(data.shape[0]*0.8):int(data.shape[0]*0.9)]
test_data = data[int(data.shape[0]*0.9):]
print("train data: {} \nvalid data:{} \ntest data:{}".format(train_data.shape, valid_data.shape, test_data.shape))

train_data = scaler.transform(train_data)
valid_data = scaler.transform(valid_data)
test_data = scaler.transform(test_data)

train_data = train_data[np.newaxis,:]
valid_data = valid_data[np.newaxis,:]
test_data = test_data[np.newaxis,:]
print("train data: {} \nvalid data:{} \ntest data:{}".format(train_data.shape, valid_data.shape, test_data.shape))




"""
Set Parameters
"""

unused_features = ['feature_1', 'feature_2']
used_features = [col for col in cluster.columns if col not in unused_features]
target_features = ['feature_0']
print(len(used_features), len(target_features))

batch_size = 64
time_step = 1 # length between each timepoint
time_int = 1 # interval between each starting point
target_delay = np.array([30, 60])

out_channels = 1024
rrcl_num = 4
rrcl_iter = [3,3,3,3]
w_filter_size = 9
p_filter_size = 4
time_point = p_filter_size**rrcl_num
forward_layers = [200, len(target_delay)]
use_batchnorm = False
split_train = False
nonlinearity = tf.nn.elu
keep_probs = [0.5,0.5, 0.5, 0.5, 0.5, 0.5] # None or an array of 1.0 values will turn off the dropout
pool=['n','p','p','p','p']
std = 0.01
l_rate = 0.0000001
l_decay = 0.1
l_step = 200*(np.max(train_data.shape)-time_point*time_step-batch_size*time_int-np.max(target_delay))/(batch_size*time_int) # 200 epochs
# optimizer = 'RMSProp' 
decay = 0.8
momentum = 0
tpa_coeff = 10
## optimizer='Adam'
##opt_epsilon = 1



"""
Set Path
"""

save_path = os.path.join('/data2/data-v0/')
path = os.path.join('data-v0')

if not os.path.exists(os.path.join('./', path)):
    os.mkdir(os.path.join('./',path))
if not os.path.exists(os.path.join(save_path,path)):
    os.mkdir(os.path.join(save_path,path))

path = os.path.join(path, 'data0')
if not os.path.exists(os.path.join('./',path)):
    os.mkdir(os.path.join('./',path))
if not os.path.exists(os.path.join(save_path,path)):
    os.mkdir(os.path.join(save_path,path))
    
path = os.path.join(path, 'target-'+ str(target_features)[1:-1].replace("'","").replace(", ","_"))
if not os.path.exists(os.path.join('./',path)):
    os.mkdir(os.path.join('./',path))
if not os.path.exists(os.path.join(save_path,path)):
    os.mkdir(os.path.join(save_path,path))

    
path = os.path.join(path, 'use_batch_norm') if use_batchnorm else os.path.join(path, 'no_batch_norm')
if not os.path.exists(os.path.join('./',path)):
    print 'creating difectory {}'.format(path)
    os.mkdir(os.path.join('./',path))
if not os.path.exists(os.path.join(save_path,path)):
    os.mkdir(os.path.join(save_path,path))
    
path = os.path.join(path, 'no_dropout') if (keep_probs == None or keep_probs == [1.0 for i in range(len(keep_probs))]) else os.path.join(path, 'dropout')
if not os.path.exists(os.path.join('./',path)):
    print 'creating difectory {}'.format(path)
    os.mkdir(os.path.join('./',path))
if not os.path.exists(os.path.join(save_path,path)):
    os.mkdir(os.path.join(save_path,path))
    
path = os.path.join(path, 'None') if nonlinearity==None else os.path.join(path, str(nonlinearity).split(" ")[1])
if not os.path.exists(os.path.join('./',path)):
    print 'creating difectory {}'.format(path)
    os.mkdir(os.path.join('./',path))      
if not os.path.exists(os.path.join(save_path,path)):
    os.mkdir(os.path.join(save_path,path))
    
model_name = 'cl6-1-'+str(target_delay)[1:-1].replace(' ','_')+'m-RRCL4_iter'+str(rrcl_iter)[1:-1].replace(', ','_')+'-'+\
             str([out_channels] + forward_layers)[1:-1].replace(', ','_')+\
             '-keep'+str(keep_probs)[1:-1].replace(', ','_').replace('0.','')+\
             '-batch128-tstep1-tint'+str(time_int)+'-std0_01-lrate'+str(l_rate)+\
             '-lstep200-l_decay0_1-decay'+str(decay).replace('.','_')+'-mom'+str(momentum).replace('.','_')

print(os.path.join(path, model_name))


if not os.path.exists(os.path.join('./', path, model_name)):
    os.mkdir(os.path.join('./', path, model_name))
    print('path created: {}'.format(os.path.join('./',path, model_name)))
if not os.path.exists(os.path.join(save_path,path, model_name)):
    os.mkdir(os.path.join(save_path, path, model_name))
    print('path created: {}'.format(os.path.join(save_path, path, model_name)))




"""
Train
"""

# train parameters
num_epochs = 200
t_loss=[]
v_loss=[]

val_freq = 1
test_freq = 20
save_freq = 50
train_history = pd.DataFrame(index=np.arange(0, num_epochs), 
                             columns=['epoch', 'loss', 'timestamp'])
valid_history = pd.DataFrame(index=np.arange(0, num_epochs/val_freq),
                             columns=['epoch', 'loss', 'timestamp'])
#val_epoch = range(1,10,1) + range(10,50,5) + range(50,100,10) + range(100, num_epochs+1, 20)
#valid_history = pd.DataFrame(index=val_epoch,
#                             columns=['epoch', 'loss', 'timestamp'])


if 'model' in globals():
    model.terminate()
model = GRCNN(batch_size = batch_size, 
                   time_point = time_point, # length.
                   in_channels = len(used_features), #number of channels of input data.
                   out_channels = out_channels, 
                   cluster = np.asarray(cluster[used_features].loc['cluster'].tolist()), #cluster index list.
                   rrcl_iter = rrcl_iter,  # number of iterations in each RRCL.
                   rrcl_num = rrcl_num, # number of RRCLs. 
                   forward_layers = forward_layers, # [(concatenated layer node omitted) forward_1, ..., forward_fin, output]
                   use_batchnorm = use_batchnorm, 
                   scale=1, offset=1e-10, epsilon=1e-10, # parameters for batch_normalization.
                   nonlinearity = nonlinearity, # nonlinearity function
                   keep_probs = keep_probs, # dropout keep_probs. 
                   # Must be in form [conv_layer, RRCL_1, ..., RRCL_fin, forward_1, ..., forward_fin], 
                   # length should be  rrcl_num + len(forward_layers)-2
                   # Use None if you don't want to use dropout.
                   pool=pool,
                   std = std,
                   w_filter_size=w_filter_size, # filter size for conv layer and RRCLs. cut into half if too long.
                   p_filter_size=p_filter_size, # max pooling filter size. p_filter_size**rrcl_num must be same as time_point.
                   l_rate=l_rate, l_decay=l_decay, l_step=l_step,
                   decay=decay, momentum=momentum,
                   tpa_coeff=tpa_coeff
                   #optimizer='Adam',
                  )



for epoch in range(num_epochs):
    loss = 0 ; 
    train_batches = 0
    start_time = time.time()
    flag=False
    for batch in utils.iterate_3d_2(inputs=np.delete(train_data, [data.columns.tolist().index(i) for i in unused_features], axis=2),
                                targets=train_data[:,:,[data.columns.tolist().index(i) for i in target_features]],
                                target_delay=np.array(target_delay), batch_size=batch_size, length = max(train_data.shape), 
                                time_point=time_point, time_step=time_step, time_int=time_int):
        train_in, train_target = batch
        train_target = train_target.reshape([batch_size,-1])
        train_batches += 1 
        loss += model.train(data=train_in, target=train_target)
        if np.isnan(loss):
            print 'ERROR!'
            flag=True
            break
    if flag:
        train_history.to_csv(os.path.join('./', path, model_name, "history_train.csv"))
        valid_history.to_csv(os.path.join('./', path, model_name, "history_valid.csv"))
        break
    t_loss.append(loss/train_batches)
    train_history.loc[epoch] = [epoch+1, t_loss[epoch], time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
    
    if(epoch+1)%val_freq ==0:
        loss = 0 ; 
        val_batches=0
        for batch in utils.iterate_3d_2(inputs=np.delete(valid_data, [data.columns.tolist().index(i) for i in unused_features], axis=2),
                                targets=valid_data[:,:,[data.columns.tolist().index(i) for i in target_features]],
                                target_delay=np.array(target_delay), batch_size=batch_size, length = max(valid_data.shape), 
                                time_point=time_point, time_step=time_step, time_int=time_int):
            val_in, val_target = batch
            val_target = val_target.reshape([batch_size,-1])
            val_batches = val_batches+1
            loss += model.test(data= val_in, target=val_target)
        v_loss.append(loss/val_batches)
        valid_history.loc[epoch/val_freq] = [epoch+1, v_loss[epoch/val_freq], time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
        if not os.path.exists(os.path.join('./', path, model_name)):
            os.mkdir( os.path.join('./', path,model_name) )
    if(epoch+1)%test_freq==0:
        test(model, save_path, path, model_name, epoch, data_columns=data.columns, target_features=target_features, 
             target_delay=target_delay, test_data=train_data, time_point=time_point, time_step=time_step, scaler=scaler, 
             xrange=10000, file_name='train_output')
    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t{:.6f}".format(t_loss[epoch]))
    if (epoch+1)%val_freq==0:
        print("  validation loss:\t{:.6f}".format(loss/val_batches))
    if (epoch+1)%save_freq==0:
        #model.save( os.path.join('./', path, model_name, str(epoch+1)+'.ckpt') )
        model.save( os.path.join(save_path, path, model_name, str(epoch+1)+'.ckpt') )
        train_history.to_csv(os.path.join('./', path, model_name, "history_train.csv"))
        valid_history.to_csv(os.path.join('./', path, model_name, "history_valid.csv"))



plt.figure(figsize=(15,5))
plt.subplot(121)

plt.plot(train_history['loss'].tolist(), label='train loss')
plt.plot( range(val_freq, len(train_history)+val_freq, val_freq), valid_history['loss'], label='valid loss', color='Red')#, marker='o'
plt.axis([0, len(train_history), 0, 2])
plt.legend(fontsize=12, bbox_to_anchor=(1.05,1),loc=2)
#plt.legend(['train loss'])#,'test loss'])#,'accuracy'])
plt.title('Loss graph', fontsize=15)
plt.xlabel('epoch', fontsize=13)
plt.ylabel('loss', fontsize=13)

plt.savefig(os.path.join('./', path, model_name, str(len(train_history))+'epochs_tvloss.png'))
print os.path.join('./', path, model_name, str(len(train_history))+'epochs_tvloss.png')
#plt.savefig(os.path.join(save_path, path, model_name, str(len(train_history))+'epochs_tvloss.png'))
#print os.path.join(save_path, path, model_name, str(len(train_history))+'epochs_tvloss.png')





