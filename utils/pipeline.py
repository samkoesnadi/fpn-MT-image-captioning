"""
Pipeline for the model to train and predict
"""

from models.transformer import *
from tqdm import tqdm

class Pipeline():
	"""
	The main class that runs shit
	"""
	def __init__(self, tokenizer_filename, checkpoint_path, max_seq_len, start_epoch_acc=0.):
		# load tokenizer
		self.tokenizer = load_tokenizer_from_path(tokenizer_filename)
		self.metric_eval = MetricEval(DATADIR, DATATYPE_VAL)

		self.max_seq_len = max_seq_len

		self.target_vocab_size = len(self.tokenizer.index_word)  # the total length of index
		input_vocab_size = math.ceil(IMAGE_INPUT_SIZE / 16) ** 2  # the input vocab size is the last dimension from Feature Extractor, i.e. if the input is 512, max input_vocab_size would be 32*32

		# instance of Transformer
		self.transformer = Transformer(num_layers, d_model, num_heads, dff,
		                               input_vocab_size, self.target_vocab_size, DROPOUT_RATE, max_seq_len=self.max_seq_len)



		# define optimizer and loss
		self.learning_rate = CustomSchedule(dff, WARM_UP_STEPS)
		self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, amsgrad=True, clipnorm=1.)

		self.loss_object_sparse = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

		# define train loss and accuracy
		self.train_loss = tf.keras.metrics.Mean(name='train_loss')

		# checkpoint
		self.ckpt = tf.train.Checkpoint(transformer=self.transformer,
		                                optimizer=self.optimizer)

		self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=100)

		self.smart_ckpt_saver = SmartCheckpointSaver(self.ckpt_manager, start_epoch_acc)

		# if a checkpoint exists, restore the latest checkpoint.
		if self.ckpt_manager.latest_checkpoint:
			self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
			print('Latest checkpoint restored!!')

	def loss(self, real, pred):
		mask = tf.math.logical_not(tf.math.equal(real, 0))
		loss_ = self.loss_object_sparse(real, pred)

		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask

		return tf.reduce_mean(loss_)

	# The @tf.function trace-compiles train_step into a TF graph for faster
	# execution. The function specializes to the precise shape of the argument
	# tensors. TODO if possible: To avoid re-tracing due to the variable sequence lengths or variable
	# batch sizes (the last batch is smaller), use input_signature to specify
	# more generic shapes.
	@tf.function
	def train_step(self, img, caption_token):
		tar_inp = caption_token[:, :-1]
		tar_real = caption_token[:, 1:]

		_mask = create_masks(tar_inp)

		with tf.GradientTape() as tape:
			predictions, _ = self.transformer(img, tar_inp,
			                                  True,
			                                  _mask)
			loss = self.loss(tar_real, predictions)

		gradients = tape.gradient(loss, self.transformer.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

		self.train_loss(loss)

	def predict(self, img, max_seq_len, plot_layer=False):
		"""

		:param plot_layer: Boolean to plot the intermediate layers
		:param img: (height, width, 3)
		:return: beam_result, attention_weights, coatt_weights ... attention_weights are from the decoder, coatt_weights from RetinaNet in the encoder
		"""
		start_token = self.tokenizer.word_index['<start>']
		end_token = self.tokenizer.word_index['<end>']

		# preprocessing
		img_expand_dims = tf.expand_dims(img, 0)
		encoder_output, coatt_weights = self.transformer.encoder(img_expand_dims, False, None)  # (batch_size, inp_seq_len, d_model)

		# For beam search, tile encoder_output
		encoder_output = tf.tile(encoder_output, tf.constant([BEAM_SEARCH_N, 1, 1]))

		# as the target is english, the first word to the transformer should be the
		# english start token.
		beam_output = tf.expand_dims([start_token] * BEAM_SEARCH_N, -1)
		beam_prob = tf.expand_dims([1] * BEAM_SEARCH_N, -1)
		beam_result = None

		for i in range(max_seq_len):
			look_ahead_mask = create_look_ahead_mask(tf.shape(beam_output)[1])

			# predictions.shape == (batch_size, seq_len, vocab_size)
			predictions, attention_weights = self.transformer(encoder_output,
			                                                  beam_output,
			                                                  False,
			                                                  look_ahead_mask)

			# select the last word from the seq_len dimension
			predictions = predictions[:, -1:, :]  # (BEAM_SEARCH_N, 1, vocab_size)

			predictions = tf.nn.softmax(predictions)  # softmax the output

			predictions = tf.reshape(predictions, [BEAM_SEARCH_N, self.target_vocab_size])

			# candidates and put it to beam_output
			candidates = predictions * tf.cast(beam_prob, tf.float32)
			candidates = tf.reshape(candidates, [-1])

			candidates_index = tf.range(self.target_vocab_size)
			candidates_index = tf.tile(tf.expand_dims(candidates_index, axis=0), [BEAM_SEARCH_N, 1])

			top_k_beams = tf.math.top_k(candidates, BEAM_SEARCH_N)
			top_k_beams_index = top_k_beams[1]
			i_beams = top_k_beams_index // self.target_vocab_size
			j_beams = top_k_beams_index - i_beams * self.target_vocab_size
			ij = tf.stack((i_beams, j_beams), axis=-1)

			a_beam = tf.gather_nd(beam_output, tf.expand_dims(ij[:, 0], axis=-1))
			b_beam = tf.gather_nd(candidates_index, tf.expand_dims(ij, axis=1))

			beam_output = tf.concat([a_beam, b_beam], axis=-1)

			# update beam probabilities
			beam_prob_pre = top_k_beams[0]
			beam_prob = tf.expand_dims(beam_prob_pre, axis=-1)

			predicted_beam_id = tf.cast(tf.argmax(beam_prob, axis=0)[0], tf.int32)  # greedily take maximum prediction
			beam_result = beam_output[predicted_beam_id]

			# return the result if the predicted_id is equal to the end token
			if beam_result[-1] == end_token:
				return beam_result[1:-1], attention_weights, coatt_weights

		# return the result if the predicted_id is equal to the end token
		if beam_result[-1] == end_token:
			return beam_result[1:-1], attention_weights, coatt_weights
		else:
			return beam_result[1:], attention_weights, coatt_weights

	def evaluate(self, generator, max_seq_len):
		"""

		:param max_seq_len:
		:param generator: dataset generator
		:return: list of result for the whole generator dataset
		"""
		results = []

		total_iter = AMOUNT_OF_VALIDATION if N_VAL_DATASET is None else N_VAL_DATASET

		for (img, imgId) in tqdm(generator, total=total_iter):
			result = self.predict(img, max_seq_len)[0]
			result = self.tokenizer.sequences_to_texts([result.numpy()])[0]
			results.append({
				"image_id": imgId,
				"caption": result
			})

		return results

	def evaluate_img(self, img, max_seq_len, imgId=0):
		"""

		:param max_seq_len:
		:param generator: dataset generator
		:return: list of result for the whole generator dataset
		"""
		results = []

		result, attention_weights, coatt_weights = self.predict(img, max_seq_len)
		result = self.tokenizer.sequences_to_texts([result.numpy()])[0]
		results.append({
			"image_id": imgId,
			"caption": result
		})

		return results, attention_weights, coatt_weights

	def plot_attention_weights(self, attention, input, caption_token, layer, filename, max_len=10):
		"""

		:param max_len: maximum length for sequence of input and sxn_result. Keep this to small value
		:param attention:
		:param input: (49)
		:param result: sxn token (seq_len_of_sxn_token)
		:param layer:
		:return:
		"""
		fig = plt.figure(figsize=(16, 8))

		attention = tf.squeeze(attention[layer], axis=0)

		# Truncate length to max_len
		attention = tf.slice(attention, [0, 0, 0], [-1, max_len, max_len])  # slice the tensor
		input = input[:max_len]
		caption_token = caption_token[:max_len]

		# temp var
		row = math.ceil(attention.shape[0] ** .5)

		for head in range(attention.shape[0]):
			ax = fig.add_subplot(row, row, head + 1)

			# plot the attention weights
			ax.matshow(attention[head][:-1, :], cmap='viridis')

			fontdict = {'fontsize': 10}

			ax.set_xticks(range(len(input)))
			ax.set_yticks(range(len(caption_token)))

			ax.set_ylim(len(caption_token) - 1.5, -0.5)

			ax.set_xticklabels(
				list(map(str, input)),
				fontdict=fontdict, rotation=90)

			ax.set_yticklabels(
				list(map(lambda i: self.tokenizer.index_word[i], caption_token)),
				fontdict=fontdict)

			ax.set_xlabel('Head {}'.format(head + 1))

		plt.tight_layout()
		plt.savefig(filename)
		plt.close()
