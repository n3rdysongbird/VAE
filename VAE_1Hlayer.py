# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 11:29:21 2018

@author: gonza
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import sklearn as skl
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import *
from scipy import signal, misc
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import matplotlib.patches as patches
import matplotlib.colors as colors
import matplotlib as mpl
import seaborn as sns

#%matplotlib inline

np.random.seed(214)
tf.set_random_seed(214)

# Load MNIST data in a format suited for tensorflow.
# The script input_data is available under this URL:
# https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
data_SPE=pd.read_excel(r'D:\JP3python\VAE\VPCR_WDC_LIQ_SPE.xlsx')
y_data = pd.read_excel(r'D:\JP3python\VAE\VPCR_WDC_LIQ_SPE_y.xlsx')
#data_SPE = data_SPE.loc[data_SPE['Container']=='Water Disp Cylinder']
data=data_SPE.iloc[:,8:]#.as_matrix()
data=data.iloc[:,490:871]
#data = np.float32(data)
locs = data_SPE['Location']
#n_samples = mnist.train.num_examples
n_samples = len(data)




###################
#Pre-Processing:
import scipy
import scipy.signal

a=pd.DataFrame(scipy.signal.savgol_filter(data,15,2,1,1.0,1))

y_2 = pd.DataFrame(columns=np.array(list(data.columns.values))[0:381])  
for index, row in a.iterrows():
    y_norm = pd.DataFrame(skl.preprocessing.normalize(row.values.reshape(1,-1), norm='l2', axis=1, copy=False, return_norm=False),columns=np.array(list(data.columns.values))[0:381])
    y_2 = y_2.append(y_norm)
    y_2.reset_index(inplace=True,drop=True)
#mean center
y_mc = y_2 - y_2.mean()
y_mc.isnull().values.any()
######################

#PLOT ORIGINAL SPECTRA AND TRANSFORMED
plt.plot(np.mean(data))
plt.plot(np.mean(y_mc))
plt.plot(np.mean(y_2))

#y_2.reset_index(inplace=True,drop=True)
#plt.figure()
#for index,row in y_2.iterrows():
#    if index % 10 == 0:    
#        plt.plot(y_2.iloc[row,:])
#    
#y_2.plot()    
#plt.plot(y_2.iloc[0,:])
#plt.plot(y_2.iloc[1,:])

######################

# PCA #
pca = skl.decomposition.PCA(n_components=2)
principalComponents = pca.fit_transform(y_mc)
pca.explained_variance_ratio_

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

principalDf['Location'] = locs.values

ax = sns.lmplot( x="principal component 1", y="principal component 2", data=principalDf, fit_reg=False, hue='Location', legend=False)
ax.set_titles('PCA (2)')
ax.set_axis_labels('PC2','PC1')
#targets = principalDf['Location'].drop_duplicates()
# Move the legend to an empty part of the plot
#plt.legend(loc='lower right')

#############################

## NEED TO COMPUTE LEVERAGE, T2, Q RESIDUALs, TO CLEAN DATA

#############################


from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(y_mc, test_size=0.2)


def next_batch(num, data):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data.reset_index(inplace=True,drop=True)
    locs.reset_index(inplace=True,drop=True)
    data_shuffle = [data.loc[i,:] for i in idx]
    labels_shuffle = [locs[i] for i in idx]

    return np.asarray(data_shuffle), labels_shuffle


def xavier_init(fan_in, fan_out, constant=1): 
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))# use constant = 4 for sigmoid, 1 for tanh activation  
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), 
                             minval=low, maxval=high, 
                             dtype=tf.float32)
    
class VariationalAutoencoder(object):
    """ Auto-Encoding Variational Bayes by Kingma and Welling """
    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus, 
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
        
        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
    
    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"], 
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1, 
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, 
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])
            
    def _initialize_weights(self, n_hidden_recog_1, 
                            n_hidden_gener_1, 
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_1, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_1, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_1, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_1, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights
            
    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']), 
                                           biases['b1'])) 
        z_mean = tf.add(tf.matmul(layer_1, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_1, weights['out_log_sigma']), 
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), 
                                           biases['b1'])) 
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['out_mean']), 
                                 biases['out_mean']))
        return x_reconstr_mean
            
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1-self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)
        #print('loss x_recon:' + str(self.x_reconstr_mean) + ' ' + str(1e-10 + 1 - self.x_reconstr_mean) )
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        ##    between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq 
                                           - tf.square(self.z_mean) 
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), 
                                  feed_dict={self.x: X})
        return cost
    
    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.z: z_mu})
    
    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean, 
                             feed_dict={self.x: X})
        
def train(datatab, network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=1):
    vae = VariationalAutoencoder(network_architecture, 
                                 learning_rate=learning_rate, 
                                 batch_size=batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = next_batch(batch_size, datatab)[0]

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            #print(cost)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        #if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), 
              "cost =", "{:.9f}".format(avg_cost))
    return vae



#vae = train(network_architecture, training_epochs=10)
#
#x_sample = next_batch(100, data_test)[0]
#from numpy import array
#x_sample = array(x_sample).reshape(1, 380)
#x_reconstruct = vae.reconstruct(x_sample)
#
#plt.figure(figsize=(8, 12))
#for i in range(5):
#
#    plt.subplot(5, 2, 2*i + 1)
#    plt.imshow(x_sample[i].reshape(1, 380), cmap="gray")
#    plt.title("Test input")
#    plt.colorbar()
#    plt.subplot(5, 2, 2*i + 2)
#    plt.imshow(x_reconstruct[i].reshape(1, 380), cmap="gray")
#    plt.title("Reconstruction")
#    plt.colorbar()
#plt.tight_layout()

n_samples = len(y_mc)

network_architecture = \
    dict(n_hidden_recog_1=15, # 1st layer encoder neurons
         n_hidden_gener_1=15, # 1st layer decoder neurons
         n_input=381,
         n_z=2)  # dimensionality of latent space

vae_2d = train(datatab=y_mc, network_architecture=network_architecture, training_epochs=21)

x_sample, y_sample = next_batch(len(y_mc), y_mc)
z_mu = vae_2d.transform(x_sample)


#y_sample dummy coding
VAEdf = pd.DataFrame(data = z_mu, columns = ['Dim 1', 'Dim 2'])
VAEdf['Location'] = y_sample  #try color by y block
VAEdf['VPCR'] = y_data['VPCR'] 

plt.figure()
ax = sns.lmplot( x="Dim 1", y="Dim 2", data=VAEdf, fit_reg=False, hue='Location', legend=True, legend_out = True)

#VPCR
plt.figure()
ax = sns.lmplot( x="Dim 1", y="Dim 2", data=VAEdf, fit_reg=False, hue='VPCR', legend=False)

#keepers = list(VAEdf[VAEdf['Dim 2']<3].index) # retroactive outliers