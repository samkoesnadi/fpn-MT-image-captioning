"""
Pipeline for the model to train and predict
"""

from models.transformer import *
from tqdm import tqdm
import tensorflow_probability as tfp
from pycocoevalcap.cider.cider import Cider

class Pipeline():
	"""
	The main class that runs shit
	"""
	def __init__(self, tokenizer_filename, checkpoint_path, max_seq_len, iter_per_epochs=None):
		# load tokenizer
		self.tokenizer = TokenTextEncoder_alphanumeric.load_from_file(tokenizer_filename)
		self.pad_token = 0
		self.end_token = self.tokenizer.encode(END_TOKEN)[0]

		self.metric_eval = MetricEval(DATADIR, DATATYPE_VAL)
		self.max_seq_len = max_seq_len

		self.target_vocab_size = self.tokenizer.num_words  # the total length of index
		input_vocab_size = math.ceil(IMAGE_INPUT_SIZE / 16) ** 2  # the input vocab size is the last dimension from Feature Extractor, i.e. if the input is 512, max input_vocab_size would be 32*32

		# define optimizer and loss
		self.learning_rate = CustomSchedule(d_model, WARM_UP_STEPS)

		with mirrored_strategy.scope():
			# instance of Transformers
			self.transformer = Transformer(num_layers, d_model, num_heads, dff,
			                               input_vocab_size, self.target_vocab_size, DROPOUT_RATE, max_seq_len=self.max_seq_len)
			self.i_transformer = Transformer(num_layers, d_model, num_heads, dff,
			                               input_vocab_size, self.target_vocab_size, DROPOUT_RATE, max_seq_len=self.max_seq_len)

			self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, amsgrad=True, beta_1=0.9, beta_2=0.98, epsilon=XE_LEARNING_EPSILON, clipnorm=1.)
			self.scst_optimizer = tf.keras.optimizers.Adam(SCST_LEARNING_RATE, amsgrad=True, beta_1=0.9, beta_2=0.98, epsilon=SCST_LEARNING_EPSILON, clipnorm=1.)

			self.loss_object_sparse = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
			self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')

			# define train loss and accuracy
			self.train_loss = tf.keras.metrics.Mean(name='train_loss')
			self.train_reward_scst_infer = tf.keras.metrics.Mean(name='cider_infer_scst_reward')
			self.train_reward_scst_train = tf.keras.metrics.Mean(name='cider_train_scst_reward')

			# checkpoint
			self.ckpt = tf.train.Checkpoint(transformer=self.transformer,
			                                optimizer=self.optimizer,
			                                scst_optimizer=self.scst_optimizer)

			self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=MAX_CKPT_TO_KEEP)

			# if a checkpoint exists, restore the latest checkpoint.
			if self.ckpt_manager.latest_checkpoint:
				checkpoint_to_restore = os.path.join(checkpoint_path, "ckpt-{}".format(CKPT_INDEX_RESTORE)) if CKPT_INDEX_RESTORE != -1 else self.ckpt_manager.latest_checkpoint
				self.ckpt.restore(checkpoint_to_restore)
				print(os.path.join(checkpoint_path, "ckpt-{}".format(CKPT_INDEX_RESTORE)) + ' checkpoint restored!!')

			# define metric for SCST
			self.cider_score_eval = Cider()

			# transfer weight
			self.i_transformer.set_weights(self.transformer.get_weights())

	def loss(self, real, pred):
		mask = tf.math.logical_not(tf.math.equal(real, 0))
		mask = tf.cast(mask, tf.float32)

		loss_ = self.loss_object_sparse(real, pred)
		loss_ *= mask

		return loss_

	def loss_scst_softmax(self, reward, predictions, masks, masks_trains):
		"""
		Policy gradient loss RL
		:param reward:
		:param pred: log_probs, shape (batch_size, max_seq_len - 1, target_vocab_size)
		:param masks_trains: its cleaning mask, no padding, no texts after end result
		:return:
		"""
		log_probs = self.loss_object(masks, predictions)  # log_softmax the prediction and mask it with the sample
		masked_log_probs = log_probs * masks_trains

		loss_ = tf.broadcast_to(tf.expand_dims(reward, -1), tf.shape(masked_log_probs)) * masked_log_probs

		return loss_

	# The @tf.function trace-compiles train_step into a TF graph for faster
	# execution. The function specializes to the precise shape of the argument
	# tensors.
	# batch sizes (the last batch is smaller), use input_signature to specify
	# more generic shapes.
	@tf.function
	def train_step(self, dist_inputs):  # this one get the input from ground truth
		def step_fn(inputs):
			img, caption_token = inputs
			tar_inp = caption_token[:, :-1]
			tar_real = caption_token[:, 1:]

			_mask = create_masks(tar_inp)

			with tf.GradientTape() as tape:
				predictions, _ = self.transformer(img, tar_inp,
				                                  True,
				                                  _mask, input_is_enc_output=False)
				loss_raw = self.loss(tar_real, predictions)
				loss = tf.reduce_sum(loss_raw) * (1.0 / XE_BATCH_SIZE)

			gradients = tape.gradient(loss, self.transformer.trainable_variables)
			self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

			self.train_loss(loss)

			return loss_raw

		per_example_losses = mirrored_strategy.experimental_run_v2(
			step_fn, args=(dist_inputs,))
		mean_loss = mirrored_strategy.reduce(
			tf.distribute.ReduceOp.SUM, per_example_losses, axis=0) / XE_BATCH_SIZE
		return mean_loss

	@tf.function
	def scst_train_step(self, dist_inputs):
		def step_fn(inputs):
			img, caption_token = inputs
			tar_inp = caption_token[:, :-1]
			tar_real = caption_token[:, 1:]

			# cast tar_real's type
			tar_real = tf.cast(tar_real, tf.int32)

			caption_token_shape = tf.shape(caption_token)  # shape of the target

			# inference prediction's result
			inf_predict_result = self.predict_batch_argmax(img, caption_token_shape[1])

			# training prediction's result
			_mask = create_masks(tar_inp)
			train_predict_result, sample_masks = self.predict_batch_sample(img, caption_token_shape[1])  # the start token is still there

			# compute loss and gradient here
			with tf.GradientTape(watch_accessed_variables=False) as tape:
				tape.watch(self.transformer.trainable_variables)
				predictions, _ = self.transformer(img, train_predict_result[:, :-1],
				                                  True,
				                                  _mask, input_is_enc_output=False)

				# cider_resInfs, cider_resTrains = self.get_scst_reward(tar_real, inf_predict_result, train_predict_result)
				cider_resInfs, cider_resTrains, masks_trains = tf.numpy_function(self.get_scst_reward, inp=[tar_real, inf_predict_result, train_predict_result[:, 1:]], Tout=[tf.float64, tf.float64, tf.float64])

				# compute loss
				reward_w_baseline = tf.cast(cider_resTrains - cider_resInfs, tf.float32)  # reward with baseline  - do not back propagate this one

				loss_raw = self.loss_scst_softmax(REWARD_DISCOUNT_FACTOR * reward_w_baseline, predictions, sample_masks, tf.cast(masks_trains, tf.float32))  # reward with baseline

				loss = tf.reduce_sum(loss_raw) * (1.0 / BATCH_SIZE)

			gradients = tape.gradient(loss, self.transformer.trainable_variables)
			self.scst_optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

			self.train_loss(loss)
			self.train_reward_scst_train(tf.reduce_mean(cider_resTrains))
			self.train_reward_scst_infer(tf.reduce_mean(cider_resInfs))

			return loss_raw

		per_example_losses = mirrored_strategy.experimental_run_v2(
			step_fn, args=(dist_inputs,))
		mean_loss = mirrored_strategy.reduce(
			tf.distribute.ReduceOp.SUM, per_example_losses, axis=0) / BATCH_SIZE
		return mean_loss


	def _clean_tokens(self, tokens):
		target_index = np.where(tokens == self.end_token)[0]
		if target_index.size != 0:
			tokens = tokens[:target_index[0] + 1]  # include the end token itself

		return tokens

	def _clean_tokens_same_len(self, tokens):
		target_index = np.where(tokens == self.end_token)[0]
		if target_index.size != 0:
			tokens[target_index[0] + 1 : ] = 0  # include the end token itself

		return tokens

	def _decode_tokens(self, tokens):
		"""
		Private function for get_scst_reward
		:param tokens: numpy array
		:return:
		"""
		# clean the tokens
		tokens = self._clean_tokens(tokens)

		# remove pad by replacing it with space
		tokens[tokens == 0] = self.pad_token

		return self.tokenizer.decode(tokens)

	def get_scst_reward(self, gts, res_infs, res_trains):
		"""
		Get the SCST's reward. CIDEr is used
		@params: in numpy
		:return: (cider_resInfs, cider_resTrains, mask_resTrains)
		"""
		# the input will be tokens, so decode those first
		gts = [self._decode_tokens(gt) for gt in gts]
		res_infs = [self._decode_tokens(res_inf) for res_inf in res_infs]

		_res_trains_original = res_trains
		res_trains = [self._decode_tokens(res_train) for res_train in res_trains]
		masks_trains = [(self._clean_tokens_same_len(res_train) != 0).astype(np.float) for res_train in _res_trains_original]  # mask all that is not padding

		# stack the batch into two, especially for CIDEr because it requires big reference
		batch_size = len(gts)
		gts = {i_batch : [gts[i_batch % batch_size]] for i_batch in range(batch_size * 2)}
		res = {i_batch : [res_infs[i_batch % batch_size]] for i_batch in range(batch_size)}  # the first batch length is the inf
		res_2  = {i_batch : [res_trains[i_batch - batch_size]] for i_batch in range(batch_size, 2 * batch_size)}

		# update res by adding res_2 to itself
		res.update(res_2)

		# do cider operation to inference and training mode of network
		_, cider_res = self.cider_score_eval.compute_score(gts, res)

		# take one batch only since we double the batch before
		cider_resInfs = cider_res[:batch_size]
		cider_resTrains = cider_res[batch_size:2*batch_size]

		return cider_resInfs, cider_resTrains, np.array(masks_trains)


	def predict_batch_sample(self, img, max_seq_len):
		"""
		Predict batch until the defined max_seq_len, this does not use BEAM_SIZE. Intended for SCST
		:param mode="sample" or "argmax"
		:return: (output, masks)... the output still has the start token, masks does not have start token
		"""

		# define start token and end token
		start_token = self.tokenizer.num_words
		softmax_temp = sample_temperature_schedule(self.scst_optimizer.iterations)
		img_shape = tf.shape(img)  # shape of the image

		# preprocessing
		encoder_output, _ = self.transformer.encoder(img, True, None)  # (batch_size, inp_seq_len, d_model)

		# first word is start token
		decoder_input = [start_token]
		output = tf.broadcast_to(decoder_input, (img_shape[0], 1))
		output = tf.concat([output, tf.zeros((img_shape[0], max_seq_len - 1), tf.int32)],
		                   axis=-1)  # add zeros in the end
		masks = tf.zeros((img_shape[0], max_seq_len - 1, self.target_vocab_size))  # minus one because there is no start token

		output_shape = tf.shape(output)  # shape of the output to inform the iteration about the output dimension
		masks_shape = tf.shape(masks)  # shape of the masks to inform the iteration about the masks dimension

		for i_seq in range(max_seq_len - 1):  # minus one because start token is already inside
			_masks = create_masks(output)

			# predictions.shape == (batch_size, seq_len, vocab_size)
			predictions, _ = self.transformer(encoder_output, output, True, _masks)

			# select the last word from the seq_len dimension
			predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

			mask = self.boltzmann_sample(predictions, temperature=softmax_temp)

			# get predicted_id from the mask
			predicted_id = tf.argmax(mask, axis=-1, output_type=tf.int32)

			# concatentate the predicted_id to the output which is given to the decoder
			# as its input.
			output = tf.concat([output[:, :i_seq + 1], predicted_id, tf.zeros([img_shape[0], max_seq_len - i_seq - 2], tf.int32)], axis=-1)
			masks = tf.concat([masks[:, :i_seq], mask, tf.zeros([img_shape[0], max_seq_len - i_seq - 2, self.target_vocab_size])], axis=1)

			output = tf.reshape(output, output_shape)
			masks = tf.reshape(masks, masks_shape)

		return output, masks

	def predict_batch_argmax(self, img, max_seq_len):
		"""
		Predict batch until the defined max_seq_len, this does not use BEAM_SIZE. Intended for SCST
		:param mode="sample" or "argmax"
		:return: (output, log_probs)
		"""

		# define start token and end token
		start_token = self.tokenizer.num_words

		img_shape = tf.shape(img)  # shape of the image

		# preprocessing
		encoder_output, _ = self.i_transformer.encoder(img, True, None)  # (batch_size, inp_seq_len, d_model)

		# first word is start token
		decoder_input = [start_token]
		output = tf.broadcast_to(decoder_input, (img_shape[0], 1))
		output = tf.concat([output, tf.zeros((img_shape[0], max_seq_len - 1), tf.int32)],
		                   axis=-1)  # add zeros in the end

		output_shape = tf.shape(output)  # shape of the output to inform the iteration about the output dimension

		for i_seq in range(max_seq_len - 1):  # minus one because start token is already inside
			_masks = create_masks(output)

			# predictions.shape == (batch_size, seq_len, vocab_size)
			predictions, _ = self.i_transformer(encoder_output, output, True, _masks)

			# select the last word from the seq_len dimension
			predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

			predicted_id = tf.argmax(predictions, axis=-1, output_type=tf.int32)

			# concatentate the predicted_id to the output which is given to the decoder
			# as its input.
			output = tf.concat([output[:, :i_seq + 1], predicted_id, tf.zeros([img_shape[0], max_seq_len - i_seq - 2], tf.int32)], axis=-1)
			output = tf.reshape(output, output_shape)

		return output[:, 1:]  # no start token inside, end token is still inside

	def boltzmann_sample(self, predictions, temperature=1.):
		"""

		:param predictions: shape(batch_size, num_words)
		:param temperature: the bigger, the more uniform it becomes. suggestion of range [1, 10]
		:return: mask. shape(batch_size, num_words)
		"""
		softmax_predictions = tf.nn.softmax(tf.math.divide(predictions, temperature))
		dist = tfp.distributions.Multinomial(total_count=1, probs=softmax_predictions)  # set distribution to sample from the predictions
		mask = dist.sample(1)[0]

		return mask


	def predict(self, img, max_seq_len):
		"""

		:param plot_layer: Boolean to plot the intermediate layers
		:param img: (height, width, 3)
		:return: beam_result, attention_weights, coatt_weights ... attention_weights are from the decoder, coatt_weights from RetinaNet in the encoder
		"""
		start_token = self.tokenizer.num_words
		end_token = self.end_token

		# preprocessing
		img_expand_dims = tf.expand_dims(img, 0)
		encoder_output, coatt_weights = self.transformer.encoder(img_expand_dims, True, None)  # (batch_size, inp_seq_len, d_model)

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
			                                                  True,
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
