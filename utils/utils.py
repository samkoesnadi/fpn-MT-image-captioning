"""
Utils function that can be used as auxiliary
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from common.common_definitions import *

import itertools

def group_list_by_delimiter(lst, delimiter):
	return [list(y) for x, y in itertools.groupby(lst, lambda z: z == delimiter) if not x]

def pad_list(lst, N, pad=0):
	lst += [pad] * (N - len(lst))
	return lst

def save_fig_png(input_arr, filename):
	"""

	:param filename:
	:param input_arr: (batch, height, width, channel)
	:return:
	"""

	input_arr = input_arr[0]  # pick the first batch
	input_arr = np.transpose(input_arr, (2,0,1))

	fig = plt.figure(figsize=(10, 10))
	len_arr = len(input_arr)

	for i, inp in enumerate(input_arr):
		ax = fig.add_subplot(math.ceil(len_arr**.5), math.ceil(len_arr**.5), i+1)  # total_x?, total_y?, index
		ax.set_title(str(inp.min())+","+str(inp.max()))  # title is min max value
		ax.imshow(inp)

	plt.savefig("layers_figure/"+filename+".png", bbox_inches="tight")
	plt.close()


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, d_model, warmup_steps=4000, multiplier=1):
		super(CustomSchedule, self).__init__()

		self.d_model = d_model
		self.d_model = tf.cast(self.d_model, tf.float32)

		self.warmup_steps = warmup_steps
		self.multiplier = multiplier

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step) * self.multiplier
		arg2 = step * (self.warmup_steps ** -1.5)

		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class CustomSchedule_rough(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, ratio_x1=5, ratio_x2=2, ratio_x3=3, ratio_y1=1, ratio_y2=.05, ratio_y3=.01, max_epoch=50,
	                 max_lr=1e-4 ):
		super(CustomSchedule_rough, self).__init__()

		self.d_model = d_model
		self.d_model = tf.cast(self.d_model, tf.float32)

		# variables for lr_scheduler
		sum_all_ratio = ratio_x1 + ratio_x2 + ratio_x3
		self.x1 = max_epoch * ratio_x1 // sum_all_ratio
		self.x2 = max_epoch * ratio_x2 // sum_all_ratio
		self.x3 = max_epoch * ratio_x3 // sum_all_ratio

		self.y1 = max_lr * ratio_y1
		self.y2 = max_lr * ratio_y2
		self.y3 = max_lr * ratio_y3

	@tf.function
	def __call__(self, step):
		x1 = self.x1
		x2 = self.x2
		x3 = self.x3

		y1 = self.y1
		y2 = self.y2
		y3 = self.y3

		if tf.math.less_equal(step, x1):
			return -(y1 / x1 ** 2) * tf.math.square(step) + 2 * y1 / x1 * step
		elif tf.math.less_equal(step, x2):
			return (y2 - y1) / (x2 - x1) * (step - x1) + y1
		else:
			return (y3 - y2) / (x3 - x2) * (step - x2) + y2


def weighted_loss(target, pred, loss_function, light_background=True):
	"""

	:param target:
	:param pred:
	:param loss_function: higher-order loss function
	:param light_background: Boolean if darker produces more attention
	:return:
	"""

	if loss_function == tf.keras.losses.MeanSquaredError:
		_loss = tf.keras.losses.MeanSquaredError(reduction="none")(target, pred)
	else:
		raise Exception("Loss function is not recognized by weighted_loss function")

	# weight the loss. TODO: change it if you want to change the ratio. now, darker thing means higher weight
	avg_pred = tf.math.reduce_mean(pred, -1)
	min_val = tf.math.reduce_min(avg_pred)
	max_val = tf.math.reduce_max(avg_pred)

	if light_background:
		ratio = (1 - (avg_pred - min_val) / (max_val - min_val)) + 1  # minus one because lighter color is less attention
	else:
		ratio = (avg_pred - min_val) / (max_val - min_val) + 1

	# _loss = tf.math.reduce_sum(ratio * _loss) / tf.math.reduce_sum(ratio)
	_loss = tf.math.reduce_sum(ratio * _loss)

	return _loss


class SmartCheckpointSaver:
	def __init__(self, ckpt_manager):
		self.ckpt_manager = ckpt_manager
		self.max_val_acc = -np.inf  # max validation accuracy
		self.max_acc_epoch = 0  # the epoch in which we have the maximum accuracy

	def __call__(self, curr_epoch, curr_val_acc):
		"""

		:param curr_epoch:
		:param curr_val_acc:
		:return: 1 ckpt saved, 0 nothing is done, -1 no new max_val_acc is created in the given rule
		"""

		# just for beginning when max_val and max_acc is empty
		if self.max_acc_epoch == 0:
			self.max_val_acc = curr_val_acc
			self.max_acc_epoch = curr_epoch

		if curr_val_acc > self.max_val_acc:
			ckpt_save_path = self.ckpt_manager.save()
			print('Saving checkpoint for epoch {} at {}'.format(curr_epoch,
			                                                    ckpt_save_path))
			self.max_val_acc = curr_val_acc
			self.max_acc_epoch = curr_epoch
			return 1
		elif curr_epoch <= MIN_EPOCH_TO_BREAK:  # if it is less or equal to MIN_EPOCH_TO_BREAK, reset everything
			self.max_val_acc = curr_val_acc
			self.max_acc_epoch = curr_epoch
		else:
			epoch_min = min(EPOCHS, max(MIN_EPOCH_TO_BREAK, int(self.max_acc_epoch * 2.)), int(self.max_acc_epoch + GAP_OF_DEAD_EPOCH))  # min epoch to break is 10

			if epoch_min <= curr_epoch:
				return -1
		return 0