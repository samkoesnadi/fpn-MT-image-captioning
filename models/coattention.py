"""
Co-attention implementation for encoder before UMV
"""

from common.common_definitions import *
from common.common_imports import *


class CoAttention_CNN(tf.keras.layers.Layer):
	def __init__(self):
		super(CoAttention_CNN, self).__init__()

	def call(self, score, hs):
		"""

		:param score: score as the attention (batch_size, width, height, 1)
		:param hs: target of attention (batch_size, width, height, num_of_channels)
		:return: co-attended matrix (batch_size, width, height, num_of_channels)
		"""

		# score and attention_weights shape == (batch_size, width, height, 1)
		# score_reshaped == (batch_size, width * height)
		score_shape = tf.shape(score)
		score_reshaped = tf.reshape(score, (score_shape[0], score_shape[1] * score_shape[2]))

		attention_weights = tf.nn.softmax(score_reshaped, axis=1)
		attention_weights = tf.reshape(attention_weights, (score_shape[0], score_shape[1], score_shape[2], score_shape[3]), name="coatt_retinanet_weights")

		# context_vector shape after == (batch_size, width, height, num_of_channels)
		context_vector = attention_weights * hs

		return context_vector, attention_weights


if __name__ == "__main__":
	if not USE_GPU:
		os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

		if tf.test.gpu_device_name():
			print('GPU found')
		else:
			print("No GPU found")

	sample_hidden = np.ones((1,7,7,1))
	sample_output = np.arange(1*7*7*3).reshape((1,7,7,3))

	sample_hidden = tf.cast(sample_hidden, tf.float32)
	sample_output = tf.cast(sample_output, tf.float32)

	attention_layer = CoAttention_CNN()
	attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

	print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
	print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))