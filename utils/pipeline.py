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
		self.tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_filename)
		self.pad_token = self.tokenizer.encode(" ")[0]
		self.end_token = self.tokenizer.encode(END_TOKEN)[0]

		self.metric_eval = MetricEval(DATADIR, DATATYPE_VAL)
		self.max_seq_len = max_seq_len

		self.target_vocab_size = self.tokenizer.vocab_size  # the total length of index
		input_vocab_size = math.ceil(IMAGE_INPUT_SIZE / 16) ** 2  # the input vocab size is the last dimension from Feature Extractor, i.e. if the input is 512, max input_vocab_size would be 32*32

		# instance of Transformers
		self.transformer = Transformer(num_layers, d_model, num_heads, dff,
		                               input_vocab_size, self.target_vocab_size, DROPOUT_RATE, max_seq_len=self.max_seq_len)

		# define optimizer and loss
		self.learning_rate = CustomSchedule(dff, WARM_UP_STEPS)
		self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, amsgrad=True, clipnorm=1.)

		self.loss_object_sparse = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

		# define train loss and accuracy
		self.train_loss = tf.keras.metrics.Mean(name='train_loss')
		self.train_loss_scst_infer = tf.keras.metrics.Mean(name='cider_infer_scst_loss')
		self.train_loss_scst_train = tf.keras.metrics.Mean(name='cider_train_scst_loss')

		# checkpoint
		self.ckpt = tf.train.Checkpoint(transformer=self.transformer,
		                                optimizer=self.optimizer)

		self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=100)

		self.smart_ckpt_saver = SmartCheckpointSaver(self.ckpt_manager, start_epoch_acc)

		# if a checkpoint exists, restore the latest checkpoint.
		if self.ckpt_manager.latest_checkpoint:
			self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
			print('Latest checkpoint restored!!')

		# define metric for SCST
		self.cider_score_eval = Cider()

	def loss(self, real, pred):
		mask = tf.math.logical_not(tf.math.equal(real, 0))
		loss_ = self.loss_object_sparse(real, pred)

		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask

		return tf.reduce_mean(loss_)

	def loss_scst_softmax(self, reward, pred):
		"""
		Policy gradient loss RL
		:param reward:
		:param pred:
		:return:
		"""
		log_prob = tf.math.log(tf.nn.softmax(pred))
		loss_ = - reward * tf.reduce_sum(tf.reshape(log_prob, (BATCH_SIZE, -1)), axis=1)

		return tf.reduce_mean(loss_)

	# The @tf.function trace-compiles train_step into a TF graph for faster
	# execution. The function specializes to the precise shape of the argument
	# tensors.
	# batch sizes (the last batch is smaller), use input_signature to specify
	# more generic shapes.
	@tf.function(input_signature=(tf.TensorSpec(shape=(XE_BATCH_SIZE, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3), dtype=tf.float32),tf.TensorSpec(shape=[XE_BATCH_SIZE, None], dtype=tf.float32)))
	def train_step(self, img, caption_token):  # this one get the input from ground truth
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

	def scst_train_step(self, img, caption_token):
		tar_inp = caption_token[:, :-1]
		tar_real = caption_token[:, 1:]

		tar_real_shape = tf.shape(tar_real)  # shape of the target

		# inference prediction's result
		inf_predict_result = self.predict_batch(img, tf.constant(tar_real_shape[1]))

		# training prediction's result
		_mask = create_masks(tar_inp)

		# compute loss and gradient here
		with tf.GradientTape() as tape:
			train_predict_result, _ = self.transformer(img, tar_inp,
			                                           True,
			                                           _mask)

			cider_resInfs, cider_resTrains = self.get_scst_reward(tar_real, inf_predict_result, train_predict_result)

			# compute loss
			loss = self.loss_scst_softmax(REWARD_DISCOUNT_FACTOR * (cider_resTrains - cider_resInfs), train_predict_result)  # reward with baseline

		gradients = tape.gradient(loss, self.transformer.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

		self.train_loss(loss)
		self.train_loss_scst_train(tf.reduce_mean(cider_resTrains))
		self.train_loss_scst_infer(tf.reduce_mean(cider_resInfs))

	def _decode_tokens(self, tokens):
		"""
		Private function for get_scst_reward
		:param tokens: numpy array
		:return:
		"""
		target_index = np.where(tokens == self.end_token)[0]
		if target_index.size != 0:
			tokens = tokens[:target_index[0] + 1]  # include the end token itself

		# remove pad by replacing it with space
		tokens[tokens == 0] = self.pad_token

		return self.tokenizer.decode(tokens.astype(np.int32))

	def get_scst_reward(self, gts, res_infs, res_trains):
		"""
		Get the SCST's reward. CIDEr is used
		:return: (cider_resInfs, cider_resTrains)
		"""
		# the input will be tokens, so decode those first
		gts = [self._decode_tokens(gt) for gt in gts.numpy()]
		res_infs = [self._decode_tokens(res_inf) for res_inf in res_infs.numpy()]

		# special handling for res_trains, find max prediction for all so that we can calculate the CIDEr
		res_trains_predicted_ids = tf.argmax(res_trains, axis=-1)
		res_trains = [self._decode_tokens(res_train) for res_train in res_trains_predicted_ids.numpy()]

		# stack the batch into two, especially for CIDEr because it requires big reference
		batch_size = len(gts)
		gts = {i_batch : [gts[i_batch % batch_size]] for i_batch in range(batch_size * 2)}
		res_infs = {i_batch : [res_infs[i_batch % batch_size]] for i_batch in range(batch_size * 2)}
		res_trains = {i_batch : [res_trains[i_batch % batch_size]] for i_batch in range(batch_size * 2)}

		# do cider operation to inference and training mode of network
		_, cider_resInfs = self.cider_score_eval.compute_score(gts, res_infs)
		_, cider_resTrains = self.cider_score_eval.compute_score(gts, res_trains)

		# take one batch only since we double the batch before
		cider_resInfs = cider_resInfs[:batch_size]
		cider_resTrains = cider_resTrains[:batch_size]

		return cider_resInfs, cider_resTrains



	@tf.function
	def predict_batch(self, img, max_seq_len):
		"""
		Predict batch until the defined max_seq_len, this does not use BEAM_SIZE. Intended for SCST
		:return:
		"""

		# define start token and end token
		start_token = self.tokenizer.vocab_size

		img_shape = tf.shape(img)  # shape of the image

		# preprocessing
		encoder_output, coatt_weights = self.transformer.encoder(img, False, None)  # (batch_size, inp_seq_len, d_model)

		# first word is start token
		decoder_input = [start_token]
		_output = tf.broadcast_to(decoder_input, (img_shape[0], 1))
		output = tf.concat([_output, tf.zeros((img_shape[0], max_seq_len + 1), tf.int32)],
		                   axis=-1)  # add zeros in the end

		for i_seq in tf.range(1, max_seq_len + 1):
			_masks = create_masks(output)

			# predictions.shape == (batch_size, seq_len, vocab_size)
			predictions, _ = self.transformer(encoder_output, output, False, _masks)

			# select the last word from the seq_len dimension
			predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

			predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

			# concatentate the predicted_id to the output which is given to the decoder
			# as its input.
			output = tf.concat([output[:, :i_seq], predicted_id, tf.zeros([img_shape[0], max_seq_len - i_seq], tf.int32)], axis=-1)

		return tf.cast(output[:, 1:], tf.int32)  # no start token inside, end token is still inside

	def predict(self, img, max_seq_len):
		"""

		:param plot_layer: Boolean to plot the intermediate layers
		:param img: (height, width, 3)
		:return: beam_result, attention_weights, coatt_weights ... attention_weights are from the decoder, coatt_weights from RetinaNet in the encoder
		"""
		start_token = self.tokenizer.vocab_size
		end_token = self.end_token

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

		total_iter = generator.max_len

		# give time limit here
		time_start = time.time()

		for (img, imgId) in tqdm(generator, total=total_iter):
			result, _, _ = self.evaluate_img(img, max_seq_len)
			results.append({
				"image_id": imgId,
				"caption": result
			})

			# if the time surpasses the limit then break the for loop
			if time.time() > time_start + MAX_EVAL_TIME:
				print("Time surparses eval time limit")
				break

		return results

	def evaluate_img(self, img, max_seq_len):
		"""

		:param max_seq_len:
		:param img: single image
		:return: list of result for the whole generator dataset
		"""

		result, attention_weights, coatt_weights = self.predict(img, max_seq_len)

		result = result.numpy()
		result[result == 0] = self.pad_token  # replace 0 with space
		result = self.tokenizer.decode(result)

		return result, attention_weights, coatt_weights
