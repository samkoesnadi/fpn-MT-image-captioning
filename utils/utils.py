"""
Utils function that can be used as auxiliary
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from common.common_definitions import *

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

def plot_att(img, attention_weights):
	"""
	GENERATOR
	TODO: for now, only take the first index of the beam
	:param img:
	:param attention_weights: dict with format ("decoder_layer{}_block_{}"). Block2: (BEAM_SIZE, num_heads, caption_seq_len, img_feature_len). Block1: (BEAM_SIZE, num_heads, caption_seq_len, caption_seq_len)
	:return: yield (caption_seq_len, num_heads, img_size, img_size, 3)
	"""
	# change range of img to 0.25..0.75, reduce img brightness so we can see saliency better
	img = (img - tf.math.reduce_min(img)) / (tf.math.reduce_max(img) - tf.math.reduce_min(img)) * .5 + .25

	# get the attention_weights shape
	att = attention_weights["decoder_layer1_block2"][0]  # get the attention (num_heads, caption_seq_len, img_feature_len)

	# max caption_seq_len = 10
	att = att[:, :MAX_SEQ_LEN_ATT_PLOT]

	att_shape = tf.cast(tf.shape(att), tf.float32)
	feature_size = tf.math.sqrt(att_shape[2])

	# broadcast image to att size
	img_broadcasted = tf.broadcast_to(img, (att_shape[1], att_shape[0], IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3))
	zeros_pyramid_level = tf.zeros((att_shape[0], att_shape[1], IMAGE_INPUT_SIZE,
	                                IMAGE_INPUT_SIZE, 2))  # this to be added when RGB is generated

	for i_layer in range(num_layers):
		att = attention_weights["decoder_layer" + str(i_layer + 1) + "_block2"][0]  # get the attention (num_heads, caption_seq_len, img_feature_len)

		# max caption_seq_len = 10
		att = att[:, :MAX_SEQ_LEN_ATT_PLOT]

		att = (att - tf.math.reduce_min(att)) / (
					tf.math.reduce_max(att) - tf.math.reduce_min(att))  # change range

		# reshape the att to square
		att = tf.reshape(att, (att_shape[0] * att_shape[1], feature_size, feature_size))  # get the attention (num_heads * caption_seq_len, feature_size, feature_size)

		# convert to RGB (red as reference for saliency)
		att = tf.expand_dims(att, -1)

		# resize att to image size
		att = tf.map_fn(lambda img: tf.image.resize(img, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE), "nearest"), att)

		# reshape it back
		att = tf.reshape(att, (att_shape[0], att_shape[1], IMAGE_INPUT_SIZE,
		                       IMAGE_INPUT_SIZE, 1))  # get the attention (num_heads, caption_seq_len, feature_size, feature_size, 1)

		# concat zeros to pyramid_level
		att = tf.concat([att, zeros_pyramid_level], -1)

		# permute the dimension to match result dimensions
		att = tf.transpose(att, perm=(1, 0, 2, 3, 4))  # (caption_seq_len, num_heads, img_size, img_size, 3)

		# change range of pyramid_level
		saliency_map = img_broadcasted + att  # add it to the image

		yield saliency_map


def plot_coatt(img, coatt_weights):
	"""
	Please make sure that img is the same size as IMAGE_INPUT_SIZE in common_definitions

	:param img:
	:param coatt_weights: (pyramid_size, batch, img_height, img_width, 1)
	:return:
	"""

	# check if coatt_weights has the right dimension
	try:
		coatt_weights_shape = tf.shape(coatt_weights[0])
		if coatt_weights_shape[0] != 1:
			raise Exception("Coatt weights do not have the right size")
	except Exception as e:
		print("Coatt_weights does not meet defined specification. Error msg:", e)
		return False

	# this to be added when RGB is generated
	zeros_pyramid_level = tf.zeros((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 2))

	# change range of img to 0.25..0.75, reduce img brightness so we can see saliency better
	img = (img - tf.math.reduce_min(img)) / (tf.math.reduce_max(img) - tf.math.reduce_min(img)) * .5 + .25

	results = []

	# iterate for every pyramid_size
	for pyramid_level in coatt_weights:
		pyramid_level = tf.squeeze(pyramid_level)  # remove all one dimensions

		pyramid_level_shape = tf.shape(pyramid_level)[0]

		# convert to RGB (red as reference for saliency)
		pyramid_level = tf.expand_dims(pyramid_level, -1)

		# resize the saliency map to fit image
		pyramid_level = (pyramid_level - tf.math.reduce_min(pyramid_level)) / (tf.math.reduce_max(pyramid_level) - tf.math.reduce_min(pyramid_level))  # change range
		pyramid_level = tf.image.resize(pyramid_level, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE), "nearest")

		# concat zeros to pyramid_level
		pyramid_level = tf.concat([pyramid_level, zeros_pyramid_level], -1)

		# change range of pyramid_level
		saliency_map = img + pyramid_level  # add it to the image

		saliency_map = (saliency_map - tf.math.reduce_min(saliency_map)) / (
					tf.math.reduce_max(saliency_map) - tf.math.reduce_min(saliency_map))  # change range

		results.append((saliency_map.numpy(), pyramid_level_shape.numpy()))

	return results

def sample_temperature_schedule(t):
	"""

	:param t: t is step
	:return: dtype float32
	"""
	# cast t to float32
	t = tf.cast(t, tf.float32)

	return tf.maximum(1., MAX_TEMPERATURE * tf.exp(-MAX_TEMPERATURE * 10 ** -4.5 * t))

# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):  # this one will go down second time some how
# 	def __init__(self, set_len):
# 		super(CustomSchedule, self).__init__()
#
# 		self.iter_per_epochs = set_len
# 		self.lr = MIN_LEARNING_RATE
#
# 	@tf.function
# 	def __call__(self, step):
# 		epoch = step // self.iter_per_epochs + 1  # starts at epoch 1
#
# 		if epoch <= 6:
# 			self.lr = tf.minimum(epoch * 1e-4, 3e-4)
# 		else:
# 			self.lr /= 2.
#
# 		return self.lr

# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):  # this one will go down second time some how
# 	def __init__(self, d_model, warmup_steps=4000, multiplier=1.):
# 		super(CustomSchedule, self).__init__()
#
# 		self.d_model = d_model
# 		self.d_model = tf.cast(self.d_model, tf.float32)
#
# 		self.warmup_steps = warmup_steps
# 		self.multiplier = multiplier
#
# 	def __call__(self, step):
# 		arg1 = tf.math.rsqrt(step) / tf.maximum((step - self.warmup_steps) * self.multiplier / (self.warmup_steps * 2), 1)
# 		arg2 = step * (self.warmup_steps ** -1.5)
#
# 		return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class SCSTCustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):  # this one will go down second time some how
	def __init__(self, max_temperature, min_temperature):
		super(SCSTCustomSchedule, self).__init__()

		if max_temperature < min_temperature:
			raise Exception("max temperature is less than min temperature")

		self.max_temperature = max_temperature
		self.min_temperature = min_temperature

		self.multiplier = 10 ** -4.5 if self.max_temperature >= 1 else 10 ** 1.5


	def __call__(self, step):
		return tf.maximum(self.min_temperature, self.max_temperature * tf.exp(-self.max_temperature * self.multiplier * step))

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
	def __init__(self, d_model, warmup_steps=4000):
		super(CustomSchedule, self).__init__()

		self.d_model = d_model
		self.d_model = tf.cast(self.d_model, tf.float32)

		self.warmup_steps = warmup_steps

	def __call__(self, step):
		arg1 = tf.math.rsqrt(step)
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


# class SmartCheckpointSaver:
# 	def __init__(self, ckpt_manager, max_val_acc=-np.inf):
# 		self.ckpt_manager = ckpt_manager
# 		self.max_val_acc = max_val_acc  # max validation accuracy
# 		self.max_acc_epoch = 0  # the epoch in which we have the maximum accuracy
#
# 	def __call__(self, curr_epoch, curr_val_acc):
# 		"""
#
# 		:param curr_epoch:
# 		:param curr_val_acc:
# 		:return: 1 ckpt saved, 0 nothing is done, -1 no new max_val_acc is created in the given rule
# 		"""
#
# 		# just for beginning when max_val and max_acc is empty
# 		if self.max_val_acc == -np.inf:
# 			self.max_val_acc = curr_val_acc
# 			self.max_acc_epoch = curr_epoch
#
# 		if curr_val_acc > self.max_val_acc:
# 			ckpt_save_path = self.ckpt_manager.save()
# 			print('Saving checkpoint for epoch {} at {}'.format(curr_epoch,
# 			                                                    ckpt_save_path))
# 			self.max_val_acc = curr_val_acc
# 			self.max_acc_epoch = curr_epoch
# 			return 1
# 		elif curr_epoch <= MIN_EPOCH_TO_BREAK:  # if it is less or equal to MIN_EPOCH_TO_BREAK, reset everything
# 			self.max_val_acc = curr_val_acc
# 			self.max_acc_epoch = curr_epoch
# 		else:
# 			epoch_min = min(EPOCHS, max(MIN_EPOCH_TO_BREAK, int(self.max_acc_epoch * 2.)), int(self.max_acc_epoch + GAP_OF_DEAD_EPOCH))  # min epoch to break is 10
#
# 			if epoch_min <= curr_epoch:
# 				return -1
# 		return 0

class TokenTextEncoder_alphanumeric():
	"""TextEncoder backed by a list of tokens.
	Tokenization splits on (and drops) non-alphanumeric characters with
	regex "\W+".
	"""
	def __init__(self,
	           captions=[],
	           oov_token="unk",
	           lowercase=True,
	            tokenizer=None):
		self._oov_token = oov_token
		self._lowercase = lowercase

		if tokenizer is None:
			self._tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=TOP_K,
																	oov_token=oov_token,
																	lower=lowercase,
																	filters='!"#$%&()*+-/:;=?@[\]^_`{|}~ ')
			self._tokenizer.fit_on_texts(captions)
		else:
			self._tokenizer = tokenizer

		# put padding in the dictionary
		self._tokenizer.word_index[''] = 0
		self._tokenizer.index_word[0] = ''

		self._vocab_size = len(self._tokenizer.word_index)

	def encode(self, s):
		# convert captions to sequences
		captions_token = self._tokenizer.texts_to_sequences([s])
		return captions_token[0]

	def decode(self, ids):
		return self._tokenizer.sequences_to_texts([ids])[0]

	@property
	def vocab_size(self):
		# Plus 1 for pad
		return self._vocab_size

	@property
	def num_words(self):
		return self._tokenizer.num_words

	@property
	def tokens(self):
		return list(self._tokenizer.index_word)

	@property
	def oov_token(self):
		return self._oov_token

	@property
	def lowercase(self):
		return self._lowercase

	@classmethod
	def _filename(cls, filename_prefix):
		return filename_prefix + ".tokens"

	def save_to_file(self, filename_prefix):
		filename = self._filename(filename_prefix)
		store_tokenizer_to_path(self._tokenizer, filename)

	@classmethod
	def load_from_file(cls, filename_prefix):
		filename = cls._filename(filename_prefix)
		_tokenizer = load_tokenizer_from_path(filename)
		_lowercase = _tokenizer.lower
		_oov_token = _tokenizer.oov_token
		return TokenTextEncoder_alphanumeric(oov_token=_oov_token,lowercase=_lowercase, tokenizer=_tokenizer)


import json

def _tokenizer_from_json(json_string):
    """Parses a JSON tokenizer configuration file and returns a
    tokenizer instance.
    # Arguments
        json_string: JSON string encoding a tokenizer configuration.
    # Returns
        A Keras Tokenizer instance
    """
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get('config')

    word_counts = json.loads(config.pop('word_counts'))
    word_docs = json.loads(config.pop('word_docs'))
    index_docs = json.loads(config.pop('index_docs'))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))

    tokenizer = tf.keras.preprocessing.text.Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word

    return tokenizer

def load_tokenizer_from_path(path):
	"""
	:param path:
	:return:
	"""
	with open(path) as f:
		data = json.load(f)
		tokenizer = _tokenizer_from_json(data)

	return tokenizer

def store_tokenizer_to_path(tokenizer, path):
	"""
	:param tokenizer: Tokenizer object to be stored
	:param path: designated path for it
	:return:
	"""
	tokenizer_json = tokenizer.to_json()
	with open(path, 'w', encoding='utf-8') as f:
		f.write(json.dumps(tokenizer_json, ensure_ascii=False))