"""
Training pipeline
"""

from common.common_imports import *
from common.common_definitions import *
from utils.pipeline import *
from dataset import *
from tqdm import tqdm

### Main training loop
if __name__ == "__main__":
	# initialize val dataset
	val_datasets = COCO_Images_ImageID(DATADIR, DATATYPE_VAL, N_VAL_DATASET)

	additional_info = load_additional_info(ADDITIONAL_FILENAME)

	# keys of additional_info needed
	key_epoch = "mt_epoch_" + os.path.basename(
		TRANSFORMER_CHECKPOINT_PATH)  # the key name in additional info for prev epoch
	key_epoch_acc = "mt_epoch_accuracy_" + os.path.basename(
		TRANSFORMER_CHECKPOINT_PATH)

	if IS_TRAINING:
		train_datasets, max_seq_len, train_set_len = get_coco_images_dataset(DATADIR, DATATYPE_TRAIN, N_TRAIN_DATASET, batch_size=XE_BATCH_SIZE)

		# get the beginning accuracy if available (for SmartCheckpointSaver)
		if key_epoch_acc in additional_info:
			start_epoch_acc = additional_info[key_epoch_acc]
		else:
			start_epoch_acc = 0.

		master = Pipeline(TOKENIZER_FILENAME, TRANSFORMER_CHECKPOINT_PATH, max_seq_len, train_set_len)  # master pipeline

		# store the max_seq_len to additional_info as for testing purpose you would not create train_datasets
		additional_info["max_seq_len"] = max_seq_len
		store_additional_info(additional_info, ADDITIONAL_FILENAME)

		# tensorboard support
		current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
		train_log_dir = 'logs/transformer/' + current_time + '/train'
		train_summary_writer = tf.summary.create_file_writer(train_log_dir)

		### Train loop
		start_epoch = 0
		if master.ckpt_manager.latest_checkpoint:
			if key_epoch in additional_info:
				start_epoch = additional_info[key_epoch]
			else:
				start_epoch = 0

		# give a gap
		print()

		# XE training
		for epoch in range(start_epoch, XE_EPOCHS):
			master.train_loss.reset_states()

			# print epoch i / n
			print("Epoch", epoch + 1, '/', EPOCHS)

			# inp -> image, tar -> html
			with tqdm(train_datasets, total=train_set_len) as t:
				for (img, caption_token) in t:
					master.train_step(img, caption_token)
					t.set_postfix(loss=master.train_loss.result().numpy())
					t.update()

			# store loss and acc to tensorboard
			with train_summary_writer.as_default():
				tf.summary.scalar('loss', master.train_loss.result(),
				                  step=epoch)  # REMEMBER: the epoch shown in the command line is epoch+1

				# save ckpt
				ckpt_save_path = "<ckpt not saved>"
				if epoch + 1 > MIN_EPOCH_TO_SAVE_CKPT:
					ckpt_save_path = master.ckpt_manager.save()
					print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
					                                                    ckpt_save_path))

				if (epoch + 1) % N_EPOCH_TO_EVALUATE == 0 and (epoch + 1) > MIN_EPOCH_TO_EVAL:
					# evaluate
					print("Evaluating...")
					results = master.evaluate(iter(val_datasets), max_seq_len)

					# save the results to file to be evaluated by COCO library
					with open(RESULT_FILE, 'w') as outfile:
						json.dump(results, outfile)

					if len(results) != 0:
						# print metric evaluation
						cider = master.metric_eval(RESULT_FILE)

						tf.summary.scalar('CIDEr', cider,
						                  step=epoch)  # REMEMBER: the epoch shown in the command line is epoch+1

						# store each epoch's checkpoint when it is more or less stable already
						ckpt_save_path = master.ckpt_manager.save()
						print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
						                                                    ckpt_save_path))

						# store last epoch step and accuracy
						additional_info[key_epoch] = epoch + 1
						additional_info[key_epoch_acc] = cider
						store_additional_info(additional_info, ADDITIONAL_FILENAME)

						# log to file
						if os.path.exists(LOGGING_FILE_PATH):
							append_write = 'a'  # append if already exists
						else:
							append_write = 'w'  # make a new file if not

						with open(LOGGING_FILE_PATH, append_write) as _log_fd:
							_log_fd.write('Saving checkpoint for epoch {} at {}. CIDEr = {}'.format(epoch + 1,
							                                                                        ckpt_save_path,
							                                                                        cider) + '\n')

			print()

		start_epoch = XE_EPOCHS if start_epoch < XE_EPOCHS else start_epoch  # if start_epoch is less than XE_EPOCHS means that it already ran the XE loop

		# SCST training
		print("Start SCST Training")

		# restart dataset with new batch
		train_datasets, max_seq_len, train_set_len = get_coco_images_dataset(DATADIR, DATATYPE_TRAIN, N_TRAIN_DATASET, batch_size=BATCH_SIZE)

		for epoch in range(start_epoch, EPOCHS):
			master.train_loss.reset_states()
			master.train_loss_scst_infer.reset_states()
			master.train_loss_scst_train.reset_states()

			# print epoch i / n
			print("Epoch", epoch + 1, '/', EPOCHS)

			# inp -> image, tar -> html
			with tqdm(train_datasets, total=train_set_len) as t:
				for (img, caption_token) in t:
					master.scst_train_step(img, caption_token)
					t.set_postfix(loss=master.train_loss.result().numpy())
					t.update()

			# store loss and acc to tensorboard
			with train_summary_writer.as_default():
				tf.summary.scalar('loss', master.train_loss.result(),
				                  step=epoch)  # REMEMBER: the epoch shown in the command line is epoch+1
				tf.summary.scalar('loss_cider_train', master.train_loss_scst_train.result(),
				                  step=epoch)  # REMEMBER: the epoch shown in the command line is epoch+1
				tf.summary.scalar('loss_cider_infer', master.train_loss_scst_infer.result(),
				                  step=epoch)  # REMEMBER: the epoch shown in the command line is epoch+1

				ckpt_save_path = "<ckpt not saved>"

				if epoch + 1 > MIN_EPOCH_TO_SAVE_CKPT:
					ckpt_save_path = master.ckpt_manager.save()
					print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
					                                                    ckpt_save_path))

				if (epoch + 1) % N_EPOCH_TO_EVALUATE == 0:
					# evaluate
					print("Evaluating...")
					results = master.evaluate(iter(val_datasets), max_seq_len)

					# save the results to file to be evaluated by COCO library
					with open(RESULT_FILE, 'w') as outfile:
						json.dump(results, outfile)

					if len(results) != 0:
						# print metric evaluation
						cider = master.metric_eval(RESULT_FILE)

						tf.summary.scalar('CIDEr', cider,
						                  step=epoch)  # REMEMBER: the epoch shown in the command line is epoch+1

						# store last epoch step and accuracy
						additional_info[key_epoch] = epoch + 1
						additional_info[key_epoch_acc] = cider
						store_additional_info(additional_info, ADDITIONAL_FILENAME)

						# log to file
						if os.path.exists(LOGGING_FILE_PATH):
							append_write = 'a'  # append if already exists
						else:
							append_write = 'w'  # make a new file if not

						with open(LOGGING_FILE_PATH, append_write) as _log_fd:
							_log_fd.write('Saving checkpoint for epoch {} at {}. CIDEr = {}'.format(epoch + 1,
					                                                    ckpt_save_path, cider) + '\n')

			print()

		print('Saving Transformer weights for epoch {}'.format(EPOCHS))
		master.ckpt.restore(master.ckpt_manager.latest_checkpoint)  # load checkpoint that was just trained to model
		master.transformer.save_weights(TRANSFORMER_WEIGHT_PATH)  # save the preprocessing weights

	else:  # NO TRAINING, just evaluation
		max_seq_len = load_additional_info(ADDITIONAL_FILENAME)["max_seq_len"]
		master = Pipeline(TOKENIZER_FILENAME, TRANSFORMER_CHECKPOINT_PATH, max_seq_len)  # master pipeline

		# evaluate
		print("Evaluating...")
		results = master.evaluate(iter(val_datasets), max_seq_len)

		# save the results to file to be evaluated by COCO library
		with open(RESULT_FILE, 'w') as outfile:
			json.dump(results, outfile)

		if len(results) != 0:
			# print metric evaluation
			cider = master.metric_eval(RESULT_FILE)