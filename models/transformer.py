"""
Transformer Seq2seq.
Put 7x7x1280 as the encoder input, and output HTML+CSS text as the decoder output

See html_SXN_parser/parser.py's comment to see more explaination related to parsing and more implementation strategy

Author: Samuel Koesnadi 2019

Attention weights naming:
decoder_layer4_block2 means 4th layer (from maximum num_layers) and second block (from the two blocks that decoder has)
"""

from datetime import datetime

from common.common_definitions import *
from utils.utils import *
from dataset import *
from models import retinanet


### POSITIONAL ENCODING
def get_angles(pos, i, d_model):
	angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
	return pos * angle_rates


def raw_positional_encoding(position, d_model):
	# there is no new dimension added here
	angle_rads = get_angles(np.arange(position)[:, np.newaxis],
	                        np.arange(d_model)[np.newaxis, :],
	                        d_model)

	# apply sin to even indices in the array; 2i
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

	# apply cos to odd indices in the array; 2i+1
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

	return tf.cast(angle_rads, dtype=tf.float32)


def positional_encoding(position, d_model):
	return raw_positional_encoding(position, d_model)[np.newaxis, ...]


def create_padding_mask(seq):
	seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

	# add extra dimensions to add the padding
	# to the attention logits.
	return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):  # smart thing going on here I should say
	mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
	return mask  # (seq_len, seq_len)


def create_masks(tar):
	# Used in the 1st attention block in the decoder.
	# It is used to pad and mask future tokens in the input received by
	# the decoder.
	look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
	dec_target_padding_mask = create_padding_mask(tar)
	combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

	return combined_mask


def scaled_dot_product_attention(q, k, v, mask):
	"""Calculate the attention weights.
	q, k, v must have matching leading dimensions.
	k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
	The mask has different shapes depending on its type(padding or look ahead)
	but it must be broadcastable for addition.

	Args:
	  q: query shape == (..., seq_len_q, depth)
	  k: key shape == (..., seq_len_k, depth)
	  v: value shape == (..., seq_len_v, depth_v)
	  mask: Float tensor with shape broadcastable
			to (..., seq_len_q, seq_len_k). Defaults to None.

	Returns:
	  output, attention_weights
	"""

	matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

	# scale matmul_qk
	dk = tf.cast(tf.shape(k)[-1], tf.float32)
	scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

	# add the mask to the scaled tensor.
	if mask is not None:
		scaled_attention_logits += (mask * -1e9)

	# softmax is normalized on the last axis (seq_len_k) so that the scores
	# add up to 1.
	attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

	output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

	return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads):
		super(MultiHeadAttention, self).__init__()
		self.num_heads = num_heads
		self.d_model = d_model

		assert d_model % self.num_heads == 0

		self.depth = d_model // self.num_heads

		self.wq = tf.keras.layers.Dense(d_model, kernel_initializer=KERNEL_INITIALIZER)
		self.wk = tf.keras.layers.Dense(d_model, kernel_initializer=KERNEL_INITIALIZER)
		self.wv = tf.keras.layers.Dense(d_model, kernel_initializer=KERNEL_INITIALIZER)

		self.dense = tf.keras.layers.Dense(d_model,
		                                   kernel_initializer=KERNEL_INITIALIZER)  # check if activation is needed here

	def split_heads(self, x, batch_size):
		"""Split the last dimension into (num_heads, depth).
		Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
		"""
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm=[0, 2, 1, 3])

	def call(self, v, k, q, mask):
		batch_size = tf.shape(q)[0]

		q = self.wq(q)  # (batch_size, seq_len, d_model)
		k = self.wk(k)  # (batch_size, seq_len, d_model)
		v = self.wv(v)  # (batch_size, seq_len, d_model)

		q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
		k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
		v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

		# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
		# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
		scaled_attention, attention_weights = scaled_dot_product_attention(
			q, k, v, mask)

		scaled_attention = tf.transpose(scaled_attention,
		                                perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

		concat_attention = tf.reshape(scaled_attention,
		                              (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

		output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

		return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(EncoderLayer, self).__init__()

		self.mhas = [MultiHeadAttention(d_model, num_heads) for _ in range(NUM_OF_PYRAMIDS-1)]

		# Point wise feed forward network
		self.ffn1 = tf.keras.layers.Dense(dff, activation=ACTIVATION,
		                                  kernel_initializer=KERNEL_INITIALIZER)  # (batch_size, seq_len, dff)
		self.ffn2 = tf.keras.layers.Dense(d_model,
		                                  kernel_initializer=KERNEL_INITIALIZER)  # (batch_size, seq_len, d_model)

		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

		self.dropout1s = [tf.keras.layers.Dropout(rate) for _ in range(NUM_OF_PYRAMIDS-1)]
		self.dropout2 = tf.keras.layers.Dropout(rate)

	def call(self, x, training, mask):
		"""

		:param x: list of views with size of NUM_OF_PYRAMIDS
		:param training:
		:param mask:
		:return:
		"""
		baseline = x[NUM_OF_PYRAMIDS-1]

		out = baseline
		# all other views
		for i in range(NUM_OF_PYRAMIDS-1):
			mha, _ = self.mhas[i](x[i], x[i], baseline, mask)
			out += self.dropout1s[i](mha, training=training)

		out1 = self.layernorm1(out)  # (batch_size, input_seq_len, d_model)

		ffn_output = self.ffn1(out1)
		ffn_output = self.ffn2(ffn_output)  # (batch_size, input_seq_len, d_model)

		ffn_output = self.dropout2(ffn_output, training=training)
		out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

		return out2


class DecoderLayer(tf.keras.layers.Layer):
	def __init__(self, d_model, num_heads, dff, rate=0.1):
		super(DecoderLayer, self).__init__()

		self.mha1 = MultiHeadAttention(d_model, num_heads)
		self.mha2 = MultiHeadAttention(d_model, num_heads)

		# Point wise feed forward network
		self.ffn1 = tf.keras.layers.Dense(dff, activation=ACTIVATION,
		                                  kernel_initializer=KERNEL_INITIALIZER)  # (batch_size, seq_len, dff)
		self.ffn2 = tf.keras.layers.Dense(d_model,
		                                  kernel_initializer=KERNEL_INITIALIZER)  # (batch_size, seq_len, d_model)

		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

		self.dropout1 = tf.keras.layers.Dropout(rate)
		self.dropout2 = tf.keras.layers.Dropout(rate)
		self.dropout3 = tf.keras.layers.Dropout(rate)

	def call(self, x, enc_output, training,
	         look_ahead_mask, padding_mask):
		# enc_output.shape == (batch_size, input_seq_len, d_model)

		attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
		attn1 = self.dropout1(attn1, training=training)
		out1 = self.layernorm1(attn1 + x)

		attn2, attn_weights_block2 = self.mha2(
			enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
		attn2 = self.dropout2(attn2, training=training)
		out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

		ffn_output = self.ffn1(out2)
		ffn_output = self.ffn2(ffn_output)  # (batch_size, target_seq_len, d_model)

		ffn_output = self.dropout3(ffn_output, training=training)
		out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

		return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
	def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
	             rate=0.1):
		super(Encoder, self).__init__()

		self.d_model = d_model
		self.num_layers = num_layers
		self.x_order = [i for i in range(NUM_OF_PYRAMIDS) if i != BASELINE_INDEX] + [BASELINE_INDEX]

		self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)

		self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
		                   for _ in range(num_layers)]

		self.feature_extractor = retinanet.FeatureExtractor(RETINANET_WEIGHT_PATH)

		self.dropout1s = [tf.keras.layers.Dropout(rate) for _ in range(NUM_OF_PYRAMIDS)]

		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	def call(self, x, training, mask):
		"""

		:param x: (Batch, height, width, 3)
		:param training:
		:param mask:
		:return:
		"""

		# extract the feature by RetinaNet. output size will be (5, batch, height, width, d_model)
		x_w_att = self.feature_extractor(x)

		# seperate between the coatt_weights and output
		x = [x_[0] for x_ in x_w_att]
		coatt_weights = [x_[1] for x_ in x_w_att]  # shape=(NUM_OF_PYRAMIDS, batch, height, width, 1)

		# put the baseline at the back
		x = [x[i] for i in self.x_order]

		# encode embedding and position
		for i_x in range(NUM_OF_PYRAMIDS):
			_x = x[i_x]
			_x_shape = tf.shape(_x)

			# define seq len from the shape of first dimension of x
			seq_len = _x_shape[1] * _x_shape[2]

			_x = tf.reshape(_x, (_x_shape[0], seq_len, _x_shape[3]))
			_x = self.layernorm1(_x)

			_x += self.pos_encoding[:, :seq_len, :]
			_x = self.dropout1s[i_x](_x, training=training)

			# set it back to x
			x[i_x] = _x

		for i in range(self.num_layers):
			x[NUM_OF_PYRAMIDS-1] = self.enc_layers[i](x, training, mask)  # change the last one

		out = x[NUM_OF_PYRAMIDS-1]

		return out, coatt_weights  # (batch_size, baseline length, d_model), (NUM_OF_PYRAMIDS, batch, height, width, 1)


class Decoder(tf.keras.layers.Layer):
	def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
	             rate=0.1, max_position=0, max_seq_len=12):
		super(Decoder, self).__init__()

		self.d_model = d_model
		self.num_layers = num_layers

		self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
		self.pos_encoding = raw_positional_encoding(max_seq_len + max_position, d_model)

		self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
		                   for _ in range(num_layers)]
		self.dropout = tf.keras.layers.Dropout(rate)

	def call(self, x, enc_output, training,
	         look_ahead_mask, padding_mask):
		seq_len = tf.shape(x)[1]
		attention_weights = {}

		x = self.embedding(x)  # (batch_size, target_seq_len, d_model)

		x += self.pos_encoding[np.newaxis, :seq_len, :]

		x = self.dropout(x, training=training)

		for i in range(self.num_layers):
			x, block1, block2 = self.dec_layers[i](x, enc_output, training,
			                                       look_ahead_mask, padding_mask)

			attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
			attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

		# x.shape == (batch_size, target_seq_len, d_model)
		return x, attention_weights


class Transformer(tf.keras.Model):
	def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
	             target_vocab_size, rate=0.1, max_position=0, max_seq_len=12):
		super(Transformer, self).__init__()

		self.tar_inp = tf.keras.layers.InputLayer((None,), sparse=True)

		self.encoder = Encoder(num_layers, d_model, num_heads, dff,
		                       input_vocab_size, rate)

		self.decoder = Decoder(num_layers, d_model, num_heads, dff,
		                       target_vocab_size + 1, rate, max_position, max_seq_len)  # target_vocab_size + 1 for the start token

		self.final_layer = tf.keras.layers.Dense(target_vocab_size, activation="linear")

	def __call__(self, inp, tar, training, look_ahead_mask):
		if training:  # IMPORTANT: if training, then preprocess the image multiple time (because of the sequence length), otherwise please preprocess the image before calling this Transformer model
			enc_output, _ = self.encoder(inp, training, None)  # (batch_size, inp_seq_len, d_model)
		else:  # this is to speed up inference time, so put the encoder preprocessed outside of the Transformer
			enc_output = inp

		# make tar sparse
		tar = self.tar_inp(tar)

		# dec_output.shape == (batch_size, tar_seq_len, d_model)
		dec_output, attention_weights = self.decoder(
			tar, enc_output, training, look_ahead_mask, None)

		final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

		return final_output, attention_weights
