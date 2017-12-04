""" This is a naive implementation of Mr.Hinton's capsule network. 
I followed this https://github.com/cedrickchee/capsule-net-pytorch pytorch implementation for reference. 
I couldn't test with reconstruction loss used in the actual network owing to it's computational complexities.
"""


import tensorflow as tf
import numpy as np
import random
import datetime
#from utils import*
import os 

batch_size=32
rout_units=8
class batchnorm():
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon  = epsilon
			self.momentum = momentum
			self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,decay=self.momentum,updates_collections=None,epsilon=self.epsilon,scale=True,is_training=train,scope=self.name)
def lrelu(x, leak=0.2):
	return tf.maximum(x,x*leak)

def conv_cond_concat(x, y):
	x_shapes = x.get_shape()
	y_shapes = y.get_shape()
	return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv(x,num_filters,kernel=5,stride=[1,2,2,1],name="conv",padding="VALID"):
	with tf.variable_scope(name):
		w=tf.get_variable('w',shape=[kernel,kernel,x.get_shape().as_list()[3], num_filters],
		    initializer=tf.truncated_normal_initializer(stddev=0.02))

		b=tf.get_variable('b',shape=[num_filters],
		    initializer=tf.constant_initializer(0.0))
		con=tf.nn.conv2d(x, w, strides=stride,padding=padding)
		return tf.reshape(tf.nn.bias_add(con, b),con.shape)

def fcn(x,num_neurons,name="fcn"):#(without batchnorm )
	with tf.variable_scope(name):

		w=tf.get_variable('w',shape=[x.get_shape().as_list()[1],num_neurons],
		    initializer=tf.truncated_normal_initializer(stddev=0.02))

		b=tf.get_variable('b',shape=[num_neurons],
		    initializer=tf.constant_initializer(0.0))
		return tf.matmul(x,w)+b

def deconv(x,output_shape,kernel=5,stride=[1,2,2,1],name="deconv"):
	with tf.variable_scope(name):
		num_filters=output_shape[-1]
		w=tf.get_variable('w',shape=[kernel,kernel, num_filters,x.get_shape().as_list()[3]],
		    initializer=tf.truncated_normal_initializer(stddev=0.02))
		b=tf.get_variable('b',shape=[num_filters],
		    initializer=tf.constant_initializer(0.0))
		decon=tf.nn.conv2d_transpose(x, w, strides=stride,output_shape=output_shape)
		return tf.reshape(tf.nn.bias_add(decon, b),decon.shape)

def load_mnist():
	data_dir = os.path.join("./data-1", "mnist")
	# data_dir="/home/satwik/Desktop/swaayatt_satwik/gan_test_Code/data /mnist"

	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000)).astype(np.float)

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000)).astype(np.float)

	trY = np.asarray(trY)
	teY = np.asarray(teY)

	X = np.concatenate((trX, teX), axis=0)
	y = np.concatenate((trY, teY), axis=0).astype(np.int)

	seed = 547
	np.random.seed(seed)
	np.random.shuffle(X)
	np.random.seed(seed)
	np.random.shuffle(y)

	y_vec = np.zeros((len(y), 10), dtype=np.float)
	for i, label in enumerate(y):
		y_vec[i,y[i]] = 1.0

	return (X/255.),y_vec

def squash(sj,dim=2):
	sj_mag_sq = tf.reduce_sum(tf.square(sj),keep_dims=True,axis=dim)
	sj_mag 	  = tf.sqrt(sj_mag_sq)
	vj 		  = (sj_mag_sq / (1.0 + sj_mag_sq)) * (sj / sj_mag)
	return vj

def routing(x):
	with tf.variable_scope("routing") as scope:
		batch_size = x.get_shape().as_list()[0]
		x 			= tf.transpose(x,perm=[0,2,1])

		x 			= tf.expand_dims(tf.stack([x]*10,axis=2),axis=4)
		w 			= tf.get_variable('w',shape=[1,1152,10,16,8],initializer=tf.truncated_normal_initializer(stddev=0.02))
		batch_weight 			= tf.concat([w]*batch_size,axis=0)
		u_hat 		= tf.matmul(batch_weight,x)
		b_ij 		= tf.constant(np.zeros([1,1152,10,1]).astype(np.float32))

		for i in range(3):
			c_ij 	= tf.nn.softmax(b_ij,dim=2)
			c_ij 	= tf.expand_dims(tf.concat([c_ij]*batch_size,axis=0),axis=4)
			s_j 	= tf.reduce_sum((c_ij * u_hat),axis=1, keep_dims=True)
			v_j 	= squash(s_j,dim=3)
			v_j1 	= tf.concat([v_j]*1152,axis=1)
			u_vj1 	= tf.reduce_mean(tf.squeeze(tf.matmul(tf.transpose(u_hat,perm=[0,1,2,4,3]),v_j1),[3]),axis=0,keep_dims=True)
			b_ij = b_ij + u_vj1
		return tf.squeeze(v_j,axis=1)



conv_list = []

def main(x):
	with tf.variable_scope("capsule_net"):

		conv1 = tf.nn.relu(conv(x,256/2,kernel=9,stride=[1,1,1,1],name="conv1_t"))

		for i in range(rout_units):

			temp = conv(conv1,32,kernel=9,stride=[1,2,2,1],name="conv1"+str(i)+"_t")
			conv_list.append(temp)
		unit = tf.stack(conv_list,axis=1)
		#print unit.get_shape().as_list(),"----------------------------------------------------"
		unit = tf.reshape(unit,[batch_size,8,-1])
		unit = squash(unit,dim=2)
		unit = routing(unit)
		return unit

x 		=	tf.placeholder(tf.float32, [batch_size, 28,28,1], name='input')
labels  	=	tf.placeholder(tf.float32, [batch_size, 10], name='labels')
X,Y 		= 	load_mnist()
with tf.device('/gpu:0'):
	outs 	=	main(x)
	v_c 	= 	tf.sqrt(tf.reduce_sum(tf.square(outs),axis=2,keep_dims=True))
	print v_c.get_shape().as_list(),"==================================================="
	zero 	=	tf.zeros(1)
	m_plus = 0.9
	m_minus = 0.1
	loss_lambda = 0.5
	max_left= 	tf.reshape(tf.maximum(m_plus - v_c, zero),[batch_size,-1])**2
	max_right= 	tf.reshape(tf.maximum(v_c 	 - m_minus, zero),[batch_size,-1])**2
	t_c = labels
	l_c = t_c * max_left + loss_lambda * (1.0 - t_c) * max_right
	loss = tf.reduce_sum(l_c,axis=1)
	t_err=tf.reduce_mean(tf.where(tf.not_equal(tf.argmax(tf.squeeze(v_c,[2,3]),1),tf.argmax(labels,1)),x=tf.zeros(batch_size,tf.float32),y=tf.ones(batch_size,tf.float32),name=None))
	t_vars			=	tf.trainable_variables()
	opt 	= tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5).minimize(loss,var_list=t_vars)
	init   = tf.global_variables_initializer()
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.15
	with tf.Session(config=config) as sess:

		sess.run(init)
		k=0
		for epoch in range(100):
			for batch_i in range(X.shape[0]//batch_size):
				batch_x = X[(batch_i)*batch_size :(batch_i+1)*batch_size ]
				batch_y = Y[(batch_i)*batch_size :(batch_i+1)*batch_size ]
				sess.run(opt,feed_dict={x:batch_x.reshape([batch_size,28,28,1]),labels:batch_y})
				Loss,accuracy = sess.run([loss,t_err],feed_dict={x:batch_x.reshape([batch_size,28,28,1]),labels:batch_y})
				print "after run ",k," loss is ",np.mean(Loss)," accuracy is ",accuracy
				k+=1



