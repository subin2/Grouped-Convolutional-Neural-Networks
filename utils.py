
import numpy as np
#import tensorflow as tf

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def iterate_2d_1(inputs, targets, length, target_delay=np.asarray([60]), batch_size=64, time_point=64, time_step=1, time_int=1, shuffle=False):
	"""
	inputs: input data of the form (time, features)
	targets: target data of the form (time, features)
	length: length of the input data. time.
	target_delay: delay of targets compared with training batch
	time_point: number of time points to be cut in training batch
	time_step: interval of each time point
	time_int: interval of each start point
	"""
	#assert len(inputs)==len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, length-time_point*time_step-batch_size*time_int-max(target_delay), batch_size*time_int):
		train = []
		target = []
		for i in range(0, batch_size*time_int, time_int):
			train.append( inputs[range(start_idx+i, start_idx+i+time_point*time_step, time_step), :] )
			target.append( targets[start_idx+i+(time_point-1)*time_step+target_delay, :] )
		train = np.asarray(train)
		target = np.asarray(target)
		yield train, target
		#return train, target

def iterate_3d_1(inputs, targets, length, target_delay=np.asarray([60]), batch_size=64, time_point=64, time_step=1, time_int=1, shuffle=False):
	"""
	inputs: input data of the form (time, axis2, axis3)
	targets: target data of the form (time, axis2, axis3)
	length: length of the input data. time.
	target_delay: delay of targets compared with training batch
	time_point: number of time points to be cut in training batch
	time_step: interval of each time point
	time_int: interval of each start point
	"""
	#assert len(inputs)==len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, length-time_point*time_step-batch_size*time_int-max(target_delay), batch_size*time_int):
		train = []
		target = []
		for i in range(0, batch_size*time_int, time_int):
			train.append( inputs[range(start_idx+i, start_idx+i+time_point*time_step, time_step), :, :] )
			target.append( targets[start_idx+i+(time_point-1)*time_step+target_delay, :, :] )
		train = np.asarray(train)
		target = np.asarray(target)
		yield train, target
		#return train, target

def iterate_3d_2(inputs, targets, length, target_delay=np.asarray([60]), batch_size=64, time_point=64, time_step=1, time_int=1, shuffle=False):
	"""
	inputs: input data of the form (axis1, time, axis3)
	targets: target data of the form (axis1, time, axis3)
	length: length of the input data. time.
	target_delay: delay of targets compared with training batch
	time_point: number of time points to be cut in training batch
	time_step: interval of each time point
	time_int: interval of each start point
	"""
	#assert len(inputs)==len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, length-time_point*time_step-batch_size*time_int-max(target_delay), batch_size*time_int):
		train = []
		target = []
		for i in range(0, batch_size*time_int, time_int):
			train.append( inputs[:, range(start_idx+i, start_idx+i+time_point*time_step, time_step), :] )
			target.append( targets[:, start_idx+i+(time_point-1)*time_step+target_delay, :] )
		train = np.asarray(train)
		target = np.asarray(target)
		yield train, target
		#return train, target

