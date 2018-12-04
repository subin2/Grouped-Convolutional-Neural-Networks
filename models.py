import numpy as np
import tensorflow as tf



class feedforward(object):
    def __init__(self, input, weight_size, weight=None, bias=None, nonlinearity=None, use_dropout=False, keep_prob=1.0,
                 use_batchnorm=False, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='feedforward_default'):
        with tf.variable_scope(name):
            self.weight = tf.Variable( tf.random_normal( weight_size, stddev=std, dtype=tf.float32) ) if weight is None else weight
            self.bias = tf.Variable( tf.random_normal( [weight_size[-1]], stddev=std, dtype=tf.float32) ) if bias is None else bias
            network = tf.nn.bias_add( tf.matmul(input, self.weight), self.bias, name=name)
            if use_batchnorm:
                batch_mean, batch_var = tf.nn.moments(network, [0])
                network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset, scale, epsilon, name=name)
            if nonlinearity != None:
                network = nonlinearity(network, name=name)
            if use_dropout:
                network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
            self.result = network
    def get_layer(self):
        return self.result
    def get_bias(self):
        return self.bias
    def get_weight(self):
        return self.weight


class conv2d(object):
    def __init__(self, input, weight_size, weight=None, bias=None, strides=[1,1,1,1], padding='SAME',
                 pool=None, pool_size=4, nonlinearity=None, use_dropout=True, keep_prob=1.0, use_batchnorm=True,
                 std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='conv2d_default'):
        with tf.variable_scope(name):
            self.weight = tf.Variable( tf.random_normal( weight_size, stddev=std, dtype=tf.float32) ) if weight is None else weight
            self.bias = tf.Variable( tf.random_normal([weight_size[-1]], stddev=std, dtype=tf.float32) ) if bias is None else bias
            network = tf.nn.bias_add( tf.nn.conv2d(input = input, filter = self.weight, strides=strides, padding=padding),
                                     self.bias, name=name)
            if use_batchnorm:
                batch_mean, batch_var = tf.nn.moments(network, [0])#,1,2])
                network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset=offset, scale=scale, variance_epsilon=epsilon, name=name)
            if nonlinearity != None:
                network = nonlinearity(network, name=name)
            if use_dropout:
                network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
            if pool=='p':
                network = tf.nn.max_pool(value=network,
                                         ksize=[1,1,pool_size,1],
                                         strides=[1,1,pool_size,1],
                                         padding='SAME')
            self.result = network
    def get_layer(self):
        return self.result
    def get_weight(self):
        return self.weight
    def get_bias(self):
        return self.bias


class res_conv2d(object):
    def __init__(self, input, weight_size, strides=[1,1,1,1], padding='SAME', pool=None, pool_size=4, nonlinearity=None,
                 use_dropout=True, keep_prob=1.0, use_batchnorm=True, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='conv2d_default'):
        with tf.variable_scope(name):
            self.weight1 = tf.Variable( tf.random_normal( weight_size, stddev=std, dtype=tf.float32) )
            self.bias1 = tf.Variable( tf.random_normal([weight_size[-1]], stddev=std, dtype=tf.float32) )
            self.weight2 = tf.Variable( tf.random_normal( weight_size, stddev=std, dtype=tf.float32) )
            self.bias2 = tf.Variable( tf.random_normal([weight_size[-1]], stddev=std, dtype=tf.float32) )
            network = tf.nn.bias_add( tf.nn.conv2d(input = input, filter = self.weight1, strides=strides, padding=padding),
                                     self.bias1, name=name)
            if use_batchnorm:
                batch_mean, batch_var = tf.nn.moments(network, [0])#,1,2])
                network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset=offset, scale=scale, variance_epsilon=epsilon, name=name)
            if nonlinearity != None:
                network = nonlinearity(network, name=name)
            #
            network = tf.nn.bias_add( tf.nn.conv2d(input = network, filter = self.weight2, strides=strides, padding=padding),
                                     self.bias2, name=name)
            network = tf.add(input, network)
            if use_batchnorm:
                batch_mean, batch_var = tf.nn.moments(network, [0])#,1,2])
                network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset=offset, scale=scale, variance_epsilon=epsilon, name=name)
            if nonlinearity != None:
                network = nonlinearity(network, name=name)
            #
            if use_dropout:
                network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
            if pool=='p':
                network = tf.nn.max_pool(value=network,
                                         ksize=[1,1,pool_size,1],
                                         strides=[1,1,pool_size,1],
                                         padding='SAME')
            self.result = network
    def get_layer(self):
        return self.result
    def get_weight(self):
        return self.weight
    def get_bias(self):
        return self.bias


class shared_depthwise_conv2d(object):
    """
    input: tensor of shape [batch, in_height, in_width, in_channels]
    weight_size: an array of the form [filter_height, filter_width, in_channels, channel_multiplier].
        Let in_channels be 1.
    returns:
        A 4D Tensor of shape [batch, out_height, out_width, in_channels * channel_multiplier].
    """
    def __init__(self, input, weight_size, strides=[1,1,1,1], padding='SAME', pool='p', pool_size=4,
					nonlinearity=None, use_dropout=True, keep_prob=1.0, use_batchnorm=True, std=0.01,
					offset=1e-10, scale=1, epsilon=1e-10, name='depthwise_conv2d_default'):
		self.pool = pool
		self.weight_size = [weight_size[0],weight_size[1],1,weight_size[3]]
		with tf.variable_scope(name):
			self.weight = tf.Variable( tf.tile(tf.reduce_mean(tf.random_normal( weight_size, stddev=std, dtype=tf.float32), axis=2, keep_dims=True),
                                               [1,1,weight_size[2],1]))
			self.bias = tf.Variable( tf.random_normal([weight_size[-1]], stddev=std, dtype=tf.float32) )
			network = tf.add( tf.nn.depthwise_conv2d(input = input, filter = self.weight, strides=strides, padding=padding),
							self.bias, name=name)
			if use_batchnorm:
				batch_mean, batch_var = tf.nn.moments(network, axes=[0])
				network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset=offset, scale=scale, variance_epsilon=epsilon, name=name)
			if nonlinearity != None:
				network = nonlinearity(network, name=name)
			if use_dropout:
				network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
			if pool=='p':
				network = tf.nn.max_pool(value=network,
                                         ksize=[1,1,pool_size,1],
                                         strides=[1,1,pool_size,1],
                                         padding='SAME')
			self.result = network
    def get_layer(self):
        return self.result
    def get_weight(self):
        return self.weight
    def get_bias(self):
        return self.bias


class depthwise_conv2d(object):
    """
    input: tensor of shape [batch, in_height, in_width, in_channels]
    weight_size: an array of the form [filter_height, filter_width, in_channels, channel_multiplier].
        Let in_channels be 1.
    returns:
        A 4D Tensor of shape [batch, out_height, out_width, in_channels * channel_multiplier].
    """
    def __init__(self, input, weight_size, strides=[1,1,1,1], padding='SAME', pool='p', pool_size=4,
					nonlinearity=None, use_dropout=True, keep_prob=1.0, use_batchnorm=True, std=0.01,
					offset=1e-10, scale=1, epsilon=1e-10, name='depthwise_conv2d_default'):
		self.pool = pool
		with tf.variable_scope(name):
			self.weight = tf.Variable( tf.random_normal( weight_size, stddev=std, dtype=tf.float32))
			self.bias = tf.Variable( tf.random_normal([weight_size[-1]*weight_size[-2]], stddev=std, dtype=tf.float32) )
			network = tf.nn.bias_add( tf.nn.depthwise_conv2d(input = input, filter = self.weight, strides=strides, padding=padding),
							self.bias, name=name)
			if use_batchnorm:
				batch_mean, batch_var = tf.nn.moments(network, axes=[0])
				network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset=offset, scale=scale, variance_epsilon=epsilon, name=name)
			if nonlinearity != None:
				network = nonlinearity(network, name=name)
			if use_dropout:
				network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
			if pool=='p':
				network = tf.nn.max_pool(value=network,
                                         ksize=[1,1,pool_size,1],
                                         strides=[1,1,pool_size,1],
                                         padding='SAME')
			self.result = network
    def get_layer(self):
        return self.result
    def get_weight(self):
        return self.weight
    def get_bias(self):
        return self.bias


class RCL(object):
	def __init__(self, input, weight_size, weight=None, biases=None, strides=[1,1,1,1], padding='SAME', pool='p', pool_size=[1,4], num_iter=3,
                 nonlinearity=None, use_dropout=True, keep_prob=1.0, use_batchnorm=True,
                 std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='RCL_default'):
		"""
			when num_iter==1, same as conv2d
		"""
		self.pool = pool
		with tf.variable_scope(name):
			self.weight = tf.Variable( tf.random_normal(weight_size, stddev=std, dtype=tf.float32)) if weight is None else weight
			self.biases = tf.Variable( tf.random_normal([weight_size[-1]], stddev=std, dtype=tf.float32)) if biases is None else biases
			"""
			rcl = tf.nn.bias_add(tf.nn.conv2d(input=input, filter=self.weight, strides=strides, padding=padding), 
                                 self.biases)
            if use_batchnorm:
                batch_mean, batch_var = tf.nn.moments(rcl, [0])#[0,1,2]
                rcl = tf.nn.batch_normalization(rcl, batch_mean, batch_var, offset, scale, epsilon)
            if nonlinearity != None:
                rcl = nonlinearity(rcl)
            network = rcl
			"""
			network = input
 		 	if num_iter == 0:
 		 		network = tf.nn.bias_add(tf.nn.conv2d(input=network, filter=self.weight, strides=strides, padding=padding),
                                         self.biases
                                        )
				if use_batchnorm:
					batch_mean, batch_var = tf.nn.moments(network, [0])#[0,1,2]
					network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset, scale, epsilon, name=name)
				if nonlinearity != None:
					network = nonlinearity(network, name=name)
 		 	else:
				for i in range(num_iter):
					#network = tf.add( rcl,
					#                 tf.nn.bias_add(tf.nn.conv2d(input=network, filter=self.weight, strides=strides, padding=padding),
					#                               self.biases
					#                               )
					#                )
					network = tf.nn.bias_add(tf.nn.conv2d(input=network, filter=self.weight, strides=strides, padding=padding),
                                         self.biases
                                        )
					if use_batchnorm:
						batch_mean, batch_var = tf.nn.moments(network, [0])#[0,1,2]
						network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset, scale, epsilon, name=name)
					if nonlinearity != None:
						network = nonlinearity(network, name=name)
					network = tf.add(input, network)
			if use_dropout:
				network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
			if pool=='c':
				#input: [batch, height, width, channel]
				#kernel: [height, width, in_channels, out_channels]
				network = conv2d(input=network,
								weight_size=[1,pool_size,weight_size[-1],weight_size[-1]],
                                 padding='VALID',
                                 nonlinearity=nonlinearity,
                                 use_dropout=use_dropout,
                                 keep_prob=keep_prob,
                                 name=name+'_convpool')
			elif pool=='p':
				network = tf.nn.max_pool(value=network,
                                         ksize=[1,pool_size[0],pool_size[1],1],
                                         strides=[1,pool_size[0],pool_size[1],1],
                                         padding='SAME')
			self.result = network
	def get_layer(self):
		if self.pool == 'c':
			return self.result.get_layer()
		return self.result
	def get_conv_layer(self):
		if self.pool != 'c':
			raise ValueError('No conv layer is used for pooling.')
		return self.pool
	def get_weight(self):
		return self.weight
	def get_biases(self):
		return self.biases

class RCL_coef(object):
	def __init__(self, input, weight_size, strides=[1,1,1,1], padding='SAME', pool='p', pool_size=[1,4], num_iter=3, nonlinearity=None,
                 use_dropout=True, keep_prob=1.0, use_batchnorm=True, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='RCL_default'):
		"""
			when num_iter==1, same as conv2d
		"""
		self.pool = pool
		with tf.variable_scope(name):
			self.weight = tf.Variable( tf.random_normal(weight_size, stddev=std, dtype=tf.float32) )
			self.biases = tf.Variable( tf.random_normal([weight_size[-1]], stddev=std, dtype=tf.float32))
			self.coef_weight = [tf.Variable(tf.random_normal([1, 1, weight_size[-2], weight_size[-1]], stddev=std, dtype=tf.float32)) for i in range(num_iter)]
			self.coef_bias = [tf.Variable(tf.random_normal([weight_size[-1]], stddev=std, dtype=tf.float32)) for i in range(num_iter)]
			"""
			rcl = tf.nn.bias_add(tf.nn.conv2d(input=input, filter=self.weight, strides=strides, padding=padding), 
                                 self.biases)
            if use_batchnorm:
                batch_mean, batch_var = tf.nn.moments(rcl, [0])#[0,1,2]
                rcl = tf.nn.batch_normalization(rcl, batch_mean, batch_var, offset, scale, epsilon)
            if nonlinearity != None:
                rcl = nonlinearity(rcl)
            network = rcl
			"""
			network = input
 		 	if num_iter == 0:
 		 		network = tf.nn.bias_add(tf.nn.conv2d(input=network, filter=self.weight, strides=strides, padding=padding),
                                         self.biases
                                        )
				if use_batchnorm:
					batch_mean, batch_var = tf.nn.moments(network, [0])#[0,1,2]
					network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset, scale, epsilon, name=name)
				if nonlinearity != None:
					network = nonlinearity(network, name=name)
 		 	else:
				for i in range(num_iter):
					network = tf.nn.bias_add(tf.nn.conv2d(input=network, filter=self.weight, strides=strides, padding=padding),
                                         self.biases
                                        )
					network = tf.nn.bias_add(tf.nn.conv2d(input=network, filter=self.coef_weight[i], strides=strides, padding=padding),
                                         self.coef_bias[i]
                                        )
					if use_batchnorm:
						batch_mean, batch_var = tf.nn.moments(network, [0])#[0,1,2]
						network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset, scale, epsilon, name=name)
					if nonlinearity != None:
						network = nonlinearity(network, name=name)
					network = tf.add(input, network)
			if use_dropout:
				network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
			if pool=='c':
				#input: [batch, height, width, channel]
				#kernel: [height, width, in_channels, out_channels]
				network = conv2d(input=network,
								weight_size=[1,pool_size,weight_size[-1],weight_size[-1]],
                                 padding='VALID',
                                 nonlinearity=nonlinearity,
                                 use_dropout=use_dropout,
                                 keep_prob=keep_prob,
                                 name=name+'_convpool')
			elif pool=='p':
				network = tf.nn.max_pool(value=network,
                                         ksize=[1,pool_size[0],pool_size[1],1],
                                         strides=[1,pool_size[0],pool_size[1],1],
                                         padding='SAME')
			self.result = network
	def get_layer(self):
		if self.pool == 'c':
			return self.result.get_layer()
		return self.result
	def get_conv_layer(self):
		if self.pool != 'c':
			raise ValueError('No conv layer is used for pooling.')
		return self.pool
	def get_weight(self):
		return self.weight
	def get_biases(self):
		return self.biases

class depthwise_RCL(object):
	def __init__(self, input, weight_size, strides=[1,1,1,1], padding='SAME', pool='p', pool_size=[1,4], num_iter=3, nonlinearity=None,
                 use_dropout=True, keep_prob=1.0, use_batchnorm=True, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='depthwise_RCL_default'):
		"""
		when num_iter==1, same as conv2d
		"""
		self.pool = pool
		with tf.variable_scope(name):
			self.weight = tf.Variable( tf.random_normal(weight_size, stddev=std, dtype=tf.float32) )
			#self.bias = tf.Variable(tf.random_normal([weight_size[-1]*weight_size[-2]], stddev=std, dtype=tf.float32))
			self.biases = [tf.Variable( tf.random_normal([weight_size[-1]*weight_size[-2]], stddev=std, dtype=tf.float32)) for i \
						   in range(num_iter+1)]
			"""
			rcl = tf.nn.bias_add(tf.nn.depthwise_conv2d(input=input, filter=self.weight, strides=strides, padding=padding), 
                                 self.biases[0])
            if use_batchnorm:
                batch_mean, batch_var = tf.nn.moments(rcl, [0])#[0,1,2]
                rcl = tf.nn.batch_normalization(rcl, batch_mean, batch_var, offset, scale, epsilon)
            if nonlinearity != None:
                rcl = nonlinearity(rcl)
            network = rcl
			"""
			network = input
			network = tf.nn.bias_add( tf.nn.depthwise_conv2d(input=network, filter=self.weight, strides=strides, padding=padding),
                                         self.biases[0])
			if use_batchnorm:
				batch_mean, batch_var = tf.nn.moments(network, [0])#[0,1,2]
				network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset, scale, epsilon, name=name)
			if nonlinearity != None:
				network = nonlinearity(network, name=name)
			for i in range(num_iter):
				#network = tf.add( rcl,
                #                 tf.nn.bias_add(tf.nn.depthwise_conv2d(input=network, filter=self.weight, strides=strides, padding=padding),
                #                               self.biases[i+1]
                #                               )
                #                )
				network = tf.nn.bias_add( tf.nn.depthwise_conv2d(input=network, filter=self.weight, strides=strides, padding=padding),
                                         self.biases[i+1])
				if use_batchnorm:
					batch_mean, batch_var = tf.nn.moments(network, [0])#[0,1,2]
					network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset, scale, epsilon, name=name)
				if nonlinearity != None:
					network = nonlinearity(network, name=name)
				network = tf.add(input, network)
			if use_dropout:
				network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
			if pool=='c':
				network = conv2d(input=network,
								weight_size=[1,pool_size, weight_size[-1]*weight_size[-2], weight_size[-1]*weight_size[-2]],
								padding='VALID',
								nonlinearity=nonlinearity,
								use_dropout=use_dropout,
								keep_prob=keep_prob,
								name=name+'_convpool')
			elif pool=='p':
				network = tf.nn.max_pool(value=network,
                                         ksize=[1,pool_size[0],pool_size[1],1],
                                         strides=[1,pool_size[0],pool_size[1],1],
                                         padding='SAME')
			self.result = network
	def get_layer(self):
		return self.result
	def get_weight(self):
		return self.weight
	def get_biases(self):
		return self.biases



class RCNN(object):
    def __init__(self, batch_size=128, time_point=1024, in_channels = 126, out_channels=256, ch_multiplier=None,
                 rrcl_iter=2, rrcl_num=4, forward_layers=[200,3], pool=['n', 'p', 'p', 'p', 'c'],
                 use_batchnorm=True, scale=1, offset=0.01, epsilon=0.01, nonlinearity=None, keep_probs=None,
                 std=0.01, w_filter_size=9, p_filter_size=4, l_rate=0.01, l_decay=0.95, l_step=1000, decay=0.9, momentum=0.9, optimizer='RMSProp', opt_epsilon=0.1):
        self.batch_size = batch_size
        self.time_point = time_point
        self.in_channels = in_channels
        self.out_channels = out_channels
        if ch_multiplier != None:
            print'\'ch_multiplier\' is depreciated. Use \'out_channels\' instead.'
            self.out_channels = ch_multiplier
        self.rrcl_iter = rrcl_iter
        self.rrcl_num = rrcl_num
        self.use_batchnorm = use_batchnorm
        self.offset = offset
        self.scale = scale
        self.epsilon = epsilon
        self.nonlinearity = nonlinearity
        self.keep_probs = keep_probs
        self.use_dropout = not (keep_probs == None or keep_probs == [1.0 for i in range(len(keep_probs))])
        if keep_probs == None:
            self.keep_probs = [1.0 for i in range(1+rrcl_num+len(forward_layers)-1)]
        if self.use_dropout and len(keep_probs) != (1 + rrcl_num + len(forward_layers)-1):
            raise ValueError('Parameter \'keep_probs\' length is wrong.')
        self.std = std
        self.w_filter_size = w_filter_size
        self.p_filter_size = p_filter_size
        self.forward_layers = [out_channels] + forward_layers
        self.pool = pool
        if len(self.pool) != rrcl_num+1:
            raise ValueError('Parameter \'pool\' length does not match with the model shape.')
        global_step = tf.Variable(0, trainable=False)
        self.l_rate = tf.train.exponential_decay(l_rate, global_step, l_step, l_decay, staircase=True)
        self.decay = decay
        self.momentum = momentum

        self.y = tf.placeholder(tf.float32, [None, self.forward_layers[-1]], name='y')
        self.x = tf.placeholder(tf.float32, [None, 1, time_point, in_channels], name='x')

        self.build_model( )

        # Define loss and optimizer, minimize the squared error
        self.cost = tf.reduce_mean(tf.pow(self.y - self.output_layer, 2))
        if optimizer=='Adam':
            self.optimizer = tf.train.AdamOptimizer(self.l_rate, epsilon=opt_epsilon).minimize(self.cost, global_step=global_step)
        else :#optimizer=='RMSProp':
            self.optimizer = tf.train.RMSPropOptimizer(self.l_rate,
                                                   decay=self.decay,
                                                   momentum=self.momentum).minimize(self.cost, global_step = global_step)

        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.session_conf = tf.ConfigProto()
        self.session_conf.gpu_options.allow_growth = True

        self.sess = tf.InteractiveSession(config=self.session_conf)
        #self.sess = tf.InteractiveSession()

        self.sess.run(init)

        self.saver = tf.train.Saver(max_to_keep=10000)

    def build_model(self):
        #self.weights, self.biases = self.init_weights()
        length = self.time_point ##length
        network = conv2d(self.x,
                         weight_size=[1, self.w_filter_size, self.in_channels, self.out_channels],
                         nonlinearity=self.nonlinearity,
                         pool=self.pool[0],
                         pool_size = self.p_filter_size,
                         use_dropout=self.use_dropout,
                         keep_prob=self.keep_probs[0],
                         use_batchnorm=self.use_batchnorm,
                         std=self.std,
                         offset=self.offset,
                         scale=self.scale,
                         epsilon=self.epsilon,
                         name='conv2d1')
        self.conv1 = network
        #output: (batch_size, 1, in_width, out_channels*in_channels)

        """
        RCL(input, filter, strides=[1,1,1,1], padding='SAME', num_iter=3, nonlinearity=None, use_dropout=True, keep_prob=1.0, 
            use_batchnorm=True, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='RCL_default'):
        """
        #networks = self.conv1.get_layer()
        self.rrcls = []
        for r in range(self.rrcl_num):
            filter_size = self.w_filter_size
            while filter_size> length:
                filter_size = filter_size/2
            network = RCL(input = network.get_layer(),
                      weight_size = [1, filter_size, self.out_channels, self.out_channels],
                      num_iter = self.rrcl_iter,
                      nonlinearity = self.nonlinearity,
                      use_dropout = self.use_dropout,
                      keep_prob = self.keep_probs[1+r],
                      use_batchnorm = self.use_batchnorm,
                      std=self.std,
                      offset=self.offset,
                      scale=self.scale,
                      epsilon=self.epsilon,
                      pool=self.pool[r+1],
                      pool_size=[1,self.p_filter_size],
                      name='RCL'+str(r))
            self.rrcls.append(network)
            length = length/self.p_filter_size
            print'rrcl{} done'.format(r),
            print'    {}'.format(network.get_layer())
        #
        network = tf.reshape(network.get_layer(), shape=[-1, self.out_channels])# * self.keep_probs[1]]) ###
        self.flatten = network
        print 'flatten to {}'.format(self.flatten)
        """
        (input, weight, nonlinearity=None, use_dropout=False, keep_prob=1.0, 
        use_batchnorm=False, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='feedforward_default')
        """

    def train(self, data, target):
        ## data: [batch, time_idx]
        ## x: [batch, in_height, in_width, in_channels]
        train_feed_dict = {self.x:data}
        train_feed_dict.update({self.y:target})
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict=train_feed_dict
                                 )
        return cost

    def test(self, data, target):
        test_feed_dict = {self.x:data}
        test_feed_dict.update({self.y:target})
        cost = self.sess.run(self.cost,
                             feed_dict=test_feed_dict
                            )
        return cost

    def reconstruct(self, data):
        recon_feed_dict = {self.x:data}
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


class RRCNN(object):
    """
    input&output:
        x : [batch_size, 1, time_point, num_features_per cluster] * number of clusters
        y : [batch, output_nodes]
    inner variables:
        x: tensorflow placeholder. input.
        y: tensorflow placeholder. target.
        sess: tensorflow session
        conv1: a list of 'conv2d' objects. use get_weight(), get_bias(), get_layer().
        rrcls: a list of list of 'RRCL' objects. first list contains RRCL per layer. second list represents the clusters.
                use get_weight(), get_biases(), get_layer().
        forwards: list of 'feedforward' classes. use get_weight(), get_bias(), get_layer().
        output: a instance of class 'feedforward'.
        output_layer: a tensorflow tensor instance. output.get_layer().
    functions:
        train(self, data, target):
            return cost
        test(self, data, target):
            return cost
        reconstruct(self, data):
            recon_feed_dict = {self.x[i]:data[:,:,:,np.where(self.cluster==i+1)[0]] for i in range(len(np.unique(self.cluster))) }
            return output
        save(self, save_path='./model.ckpt'): save model
        load(self, load_path = './model.ckpt'): restore model
        terminate(self): close session ane reset dafault graph
    parameters:
        batch_size=128,
        time_point=256, # length.
        in_channels = train_data.shape[2], #number of channels of input data.
        out_channels=512,
        cluster=cluster, #cluster index list. should start from 0.
        rrcl_iter=3,  # number of iterations in each RRCL.
        rrcl_num=4, # number of RRCLs.
        forward_layers=[100,3], # [(concatenated layer node omitted) forward_1, ..., forward_fin, output]
        use_batchnorm=True,
        scale=1, offset=1e-10, epsilon=1e-10, # parameters for batch_normalization.
        nonlinearity=tf.nn.elu, # nonlinearity function
        keep_probs=[1.0, 1.0, 1.0, 1.0, 1.0, 0.9, ], # dropout keep_probs.
                        # Must be in form [conv_layer, RRCL_1, ..., RRCL_fin, forward_1, ..., forward_fin],
                        # length should be  rrcl_num + len(forward_layers)-2
                        # Use None if you don't want to use dropout.
        std=0.001,
        w_filter_size=9, # filter size for conv layer and RRCLs. cut in half if too long.
        p_filter_size=4, # max pooling filter size. p_filter_size**rrcl_num must be same as time_point.
        l_rate=0.01,
        l_decay=0.95,
        l_step=1000,
        decay=0.9,
        momentum=0.9
    """

    def __init__(self, batch_size=128, time_point=1024, in_channels=126, out_channels=256, ch_multiplier=None,
                 cluster=None, rrcl_iter=[2, 2, 2], rrcl_num=3, forward_layers=[200, 3], pool=['n', 'p', 'p', 'p'],
                 use_batchnorm=True, scale=1, offset=0.01, epsilon=0.01, nonlinearity=None, keep_probs=None,
                 std=0.01, w_filter_size=9, p_filter_size=4, l_rate=0.01, l_decay=0.95, l_step=1000,
                 optimizer='RMSProp', opt_epsilon=0.1, decay=0.9, momentum=0.9):
        self.batch_size = batch_size
        self.time_point = time_point
        self.in_channels = in_channels
        # self.out_channels= ch_multiplier
        self.out_channels = out_channels
        if ch_multiplier != None:
            print
            '\'ch_multiplier\' is depreciated. Use \'out_channels\''
            self.out_channels = ch_multiplier
        self.cluster = cluster
        self.rrcl_iter = rrcl_iter
        self.rrcl_num = rrcl_num
        self.use_batchnorm = use_batchnorm
        self.offset = offset
        self.scale = scale
        self.epsilon = epsilon
        self.nonlinearity = nonlinearity
        self.keep_probs = keep_probs
        self.use_dropout = not (keep_probs == None or keep_probs == [1.0 for i in range(len(keep_probs))])
        if keep_probs == None:
            self.keep_probs = [1.0 for i in range(1 + rrcl_num + len(forward_layers) - 1)]
        if self.use_dropout and len(keep_probs) != (1 + rrcl_num + len(forward_layers) - 1):
            raise ValueError('\'keep_probs\' length is wrong')
        self.std = std
        self.w_filter_size = w_filter_size
        self.p_filter_size = p_filter_size
        t = 0
        for i in range(len(np.unique(self.cluster))):
            t = t + self.out_channels * np.sum(self.cluster == i) / self.in_channels
        self.ch_sum = t
        self.forward_layers = [t] + forward_layers  ################
        self.pool = pool
        if len(self.pool) != rrcl_num + 1:
            raise ValueError('Parameter \'pool\' length does not match with the model shape.')
        global_step = tf.Variable(0, trainable=False)
        self.l_rate = tf.train.exponential_decay(l_rate, global_step, l_step, l_decay, staircase=True)
        self.decay = decay
        self.momentum = momentum
        # self.h_nums = h_nums

        self.y = tf.placeholder(tf.float32, [None, self.forward_layers[-1]], name='y');
        self.x = [tf.placeholder(tf.float32, [None, 1, time_point, np.sum(cluster == i)], name='x' + str(i)) for i
                  in range(len(np.unique(cluster)))]

        self.build_model()

        # Define loss and optimizer, minimize the squared error
        # self.cost = tf.reduce_mean(tf.pow(self.y - self.output, 2))
        # self.cost = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(self.output), reduction_indices=[1]))
        self.cost = tf.reduce_mean(tf.pow(self.y - self.output_layer, 2))
        if optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.l_rate, epsilon=opt_epsilon).minimize(self.cost,
                                                                                               global_step=global_step)
        else:  # optimizer=='RMSProp':
            self.optimizer = tf.train.RMSPropOptimizer(self.l_rate,
                                                       decay=self.decay,
                                                       momentum=self.momentum).minimize(self.cost,
                                                                                        global_step=global_step)
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.session_conf = tf.ConfigProto()
        self.session_conf.gpu_options.allow_growth = True

        self.sess = tf.InteractiveSession(config=self.session_conf)
        # self.sess = tf.InteractiveSession()

        self.sess.run(init)

        self.saver = tf.train.Saver(max_to_keep=10000)

    def build_model(self):
        # self.weights, self.biases = self.init_weights()
        length = self.time_point  ##length
        self.conv1 = []
        networks = []
        for i in range(len(np.unique(self.cluster))):
            """
            conv2d(input, filter, strides=[1,1,1,1], padding='SAME', nonlinearity=None, use_dropout=True, keep_prob=1.0, 
                   use_batchnorm=True, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='conv2d_default'):
            """
            # print i
            # print self.x[i],
            # print [1, self.w_filter_size, np.sum(self.cluster==i), self.out_channels*np.sum(self.cluster==i)/self.in_channels]
            conv1 = conv2d(self.x[i],
                           weight_size=[1, self.w_filter_size, np.sum(self.cluster == i),
                                        self.out_channels * np.sum(self.cluster == i) / self.in_channels],
                           nonlinearity=self.nonlinearity,
                           pool=self.pool[0],
                           pool_size=self.p_filter_size,
                           use_dropout=self.use_dropout,
                           keep_prob=self.keep_probs[0],
                           use_batchnorm=self.use_batchnorm,
                           std=self.std,
                           offset=self.offset,
                           scale=self.scale,
                           epsilon=self.epsilon,
                           name='conv2d_cluster' + str(i))
            self.conv1.append(conv1)
            networks.append(conv1)
            # print conv1.get_layer()
        # (batch, time, in_ch, ch_mult)
        print
        'conv done'
        """
        self.conv1p = tf.nn.max_pool(value=self.conv1, 
                                    ksize=[1,1,4,1], 
                                    strides=[1,1,4,1], 
                                    padding='SAME')
        """
        # output: (batch_size, 1, in_width, out_channels*in_channels)

        """
        RCL(input, filter, strides=[1,1,1,1], padding='SAME', num_iter=3, nonlinearity=None, use_dropout=True, keep_prob=1.0, 
            use_batchnorm=True, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='RCL_default'):
        """
        # networks = self.conv1.get_layer()
        self.rrcls = []
        for r in range(self.rrcl_num):
            rrcl = []
            filter_size = self.w_filter_size
            while filter_size > length:
                filter_size = filter_size / 2
            for i in range(len(np.unique(self.cluster))):
                # print ' cluster{} start'.format(i),
                tmp = RCL(input=networks[i].get_layer(),
                          weight_size=[1, filter_size, self.out_channels * np.sum(self.cluster == i) / self.in_channels,
                                       self.out_channels * np.sum(self.cluster == i) / self.in_channels],
                          num_iter=self.rrcl_iter[r],
                          nonlinearity=self.nonlinearity,
                          use_dropout=self.use_dropout,
                          keep_prob=self.keep_probs[1 + r],
                          use_batchnorm=self.use_batchnorm,
                          std=self.std,
                          offset=self.offset,
                          scale=self.scale,
                          epsilon=self.epsilon,
                          pool=self.pool[r + 1],
                          pool_size=[1, self.p_filter_size],
                          name='RCL' + str(r) + '_cluster' + str(i))
                rrcl.append(tmp)
                # print 'done'
            networks = rrcl
            self.rrcls.append(rrcl)
            length = length / self.p_filter_size
            print
            'rrcl{} done'.format(r),
            print
            '    {}'.format(rrcl[-1].get_layer())
        #
        networks = []
        for i in range(len(rrcl)):
            networks.append(rrcl[i].get_layer())
            # print networks[i]
        self.concat = tf.concat(3, networks)
        print
        'concatenated to {}'.format(self.concat)

        network = tf.reshape(self.concat, shape=[-1, self.ch_sum])  # * self.keep_probs[1]]) ###
        self.flatten = network
        print
        'flatten to {}'.format(self.flatten)
        """
        (input, weight, nonlinearity=None, use_dropout=False, keep_prob=1.0, 
        use_batchnorm=False, std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='feedforward_default')
        """
        if len(self.forward_layers) == 2:
            network = feedforward(input=network,
                                  weight_size=[self.forward_layers[0], self.forward_layers[1]],
                                  nonlinearity=None,
                                  use_dropout=False,
                                  use_batchnorm=False,
                                  std=self.std,
                                  offset=self.offset,
                                  scale=self.scale,
                                  epsilon=self.epsilon,
                                  name='output')
            self.output = network  # .get_layer()
            self.output_layer = network.get_layer()
            print
            'feedforward {} done, {}'.format(i + 1, self.output_layer)
            print
            'model built'
        else:
            self.forwards = []
            for i in range(len(self.forward_layers) - 1 - 1):
                network = feedforward(input=network,
                                      weight_size=[self.forward_layers[i], self.forward_layers[i + 1]],
                                      nonlinearity=self.nonlinearity,
                                      use_dropout=self.use_dropout,
                                      keep_prob=self.keep_probs[1 + r],
                                      use_batchnorm=self.use_batchnorm,
                                      std=self.std,
                                      offset=self.offset,
                                      scale=self.scale,
                                      epsilon=self.epsilon,
                                      name='forward' + str(i))
                self.forwards.append(network)
                network = network.get_layer()
                print
                'feedforward {} done, {}'.format(i, network)
            network = feedforward(input=network,
                                  weight_size=[self.forward_layers[-2], self.forward_layers[-1]],
                                  nonlinearity=None,
                                  use_dropout=False,
                                  use_batchnorm=False,
                                  std=self.std,
                                  offset=self.offset,
                                  scale=self.scale,
                                  epsilon=self.epsilon,
                                  name='output')
            self.output = network  # .get_layer()
            self.output_layer = network.get_layer()
            print
            'feedforward {} done, {}'.format(i + 1, self.output_layer)
            print
            'model built'

    def train(self, data, target):
        ## data: [batch, time_idx]
        ## x: [batch, in_height, in_width, in_channels]
        train_feed_dict = {self.x[i]: data[:, :, :, np.where(self.cluster == i)[0]] for i in
                           range(len(np.unique(self.cluster)))}
        train_feed_dict.update({self.y: target})
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict=train_feed_dict
                                  )
        return cost

    def test(self, data, target):
        test_feed_dict = {self.x[i]: data[:, :, :, np.where(self.cluster == i)[0]] for i in
                          range(len(np.unique(self.cluster)))}
        test_feed_dict.update({self.y: target})
        cost = self.sess.run(self.cost,
                             feed_dict=test_feed_dict
                             )
        return cost

    def reconstruct(self, data):
        recon_feed_dict = {self.x[i]: data[:, :, :, np.where(self.cluster == i)[0]] for i in
                           range(len(np.unique(self.cluster)))}
        return self.sess.run(self.output_layer,
                             feed_dict=recon_feed_dict
                             )

    def save(self, save_path='./model.ckpt'):
        saved_path = self.saver.save(self.sess, save_path)
        print("Model saved in file: %s" % saved_path)

    def load(self, load_path='./model.ckpt'):
        self.saver.restore(self.sess, load_path)
        print("Model restored")

    def terminate(self):
        self.sess.close()
        tf.reset_default_graph()




class LSTM(object):
	def __init__(self, std=0.01, batch_size=64, lstm_time=100, lstm_layers=[442,442,442], layers=[442,200,50,3], num_sensors=442,
			scale=1, offset=0.01, epsilon=0.01, keep_probs=[0.9, 0.8, 0.7],
			l_rate=0.01, l_decay=0.95, l_step=1000, decay=0.9, momentum=0.9):
		self.std = std
		self.batch_size = batch_size
		self.lstm_time = lstm_time
		self.lstm_layers = lstm_layers
		self.layers = layers
		self.num_sensors = num_sensors
		self.scale = scale
		self.offset = offset
		self.epsilon = epsilon
		self.keep_probs = keep_probs
		#
		global_step = tf.Variable(0, trainable=False)
		self.l_rate = tf.train.exponential_decay(l_rate, global_step, l_step, l_decay, staircase=True)
		self.decay = decay
		self.momentum = momentum
		#
		self.x = tf.placeholder(tf.float32, [None, lstm_time, lstm_layers[0]], name='x')
		self.y = tf.placeholder(tf.float32, [None, 3], name='y')
		self.build_model( )

		# Define loss and optimizer, minimize the squared error
		self.cost = tf.reduce_mean(tf.pow(self.y - self.output, 2))
		self.optimizer = tf.train.RMSPropOptimizer(self.l_rate, decay=self.decay, momentum=self.momentum).minimize(self.cost, global_step = global_step)

		# Initializing the tensor flow variables
		init = tf.initialize_all_variables()

		# Launch the session
		self.session_conf = tf.ConfigProto()
		self.session_conf.gpu_options.allow_growth = True

		self.sess = tf.InteractiveSession(config=self.session_conf)
		#self.sess = tf.InteractiveSession()

		self.sess.run(init)

		self.saver = tf.train.Saver(max_to_keep=10000)

	def build_model(self):
		#self.weights, self.biases = self.init_weights()
		"""
		dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None, dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None)
		cell: An instance of RNNCell.
		inputs: The RNN inputs.
            If time_major == False (default), this must be a Tensor of shape: [batch_size, max_time, ...], or a nested tuple of such elements.
            If time_major == True, this must be a Tensor of shape: [max_time, batch_size, ...], or a nested tuple of such elements.
		Returns: A pair (outputs, state) where:
            outputs: The RNN output Tensor
                If time_major == False (default), this will be a Tensor shaped: [batch_size, max_time, cell.output_size].
                If time_major == True, this will be a Tensor shaped: [max_time, batch_size, cell.output_size].
                Note, if cell.output_size is a (possibly nested) tuple of integers or TensorShape objects,
                then outputs will be a tuple having the same structure as cell.output_size,
                containing Tensors having shapes corresponding to the shape data in cell.output_size.
            state: The final state.
                If cell.state_size is an int, this will be shaped [batch_size, cell.state_size].
                If it is a TensorShape, this will be shaped [batch_size] + cell.state_size.
                If it is a (possibly nested) tuple of ints or TensorShape, this will be a tuple having the corresponding shapes.
		"""

		network = self.x
		self.layers = []
		for i in range(len(self.lstm_layers)-1):
			with tf.variable_scope('lstm'+str(i)):
				cells = tf.nn.rnn_cell.DropoutWrapper( tf.nn.rnn_cell.BasicLSTMCell(self.lstm_layers[i+1], state_is_tuple=True),
				self.keep_probs[i])
				outputs, states = tf.nn.dynamic_rnn(cells, network, dtype=tf.float32)
				network = outputs
				self.layers.append(network)
		#network outputs: [batch_size, max_time, cell.output_size]
		network = network[:, network.get_shape().as_list()[1]-1, :]
		batch_mean, batch_var = tf.nn.moments(network, axes=[0])
		network = tf.nn.batch_normalization(network, batch_mean, batch_var, self.offset, self.scale, self.epsilon)
		self.layers.append(network)

		for i in range(len(self.layers)-2):
			network = tf.nn.bias_add( tf.matmul(network, self.weights[i]), self.biases[i])
			batch_mean, batch_var = tf.nn.moments(network, axes=[0])
			network = tf.nn.batch_normalization(network, batch_mean, batch_var, self.offset, self.scale, self.epsilon)
			network = tf.nn.dropout( network, self.keep_probs[ (len(self.lstm_layers)-1) + i] )
			self.layers.append( network )
			self.output = tf.nn.bias_add( tf.matmul(network, self.weights[-1]),
										self.biases[-1])

	def init_weights(self):
		weights = []
		biases = []
		for i in range(len(self.layers)-1):
			weights.append( tf.Variable(tf.random_normal([layers[i], layers[i+1]], stddev=0.01, dtype=tf.float32)) )
			biases.append( tf.Variable(tf.random_normal([layers[i+1]], stddev=0.01, dtype=tf.float32)) )
		return weights, biases

	def train(self, data, target):
		## data: [batch, time_idx]
		## x: [batch, in_height, in_width, in_channels]
		opt, cost = self.sess.run((self.optimizer, self.cost),
								feed_dict={ self.y: target,
											self.x:data
											}
								)
		return cost

	def test(self, data, target):
		cost = self.sess.run(self.cost,
							feed_dict={self.y: target,
									   self.x:data
									   }
							)
		return cost

	def reconstruct(self, data):
		return self.sess.run(self.output,
							feed_dict={self.x:data}
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

