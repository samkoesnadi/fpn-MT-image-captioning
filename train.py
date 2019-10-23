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

	key_epoch = "mt_epoch_" + os.path.basename(
		TRANSFORMER_CHECKPOINT_PATH)  # the key name in additional info for prev epoch

	if IS_TRAINING:
		train_datasets, max_seq_len, train_set_len = get_coco_images_dataset(DATADIR, DATATYPE_TRAIN, N_TRAIN_DATASET)

		master = Pipeline(TOKENIZER_FILENAME, TRANSFORMER_CHECKPOINT_PATH, max_seq_len)  # master pipeline

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
				start_epoch = additional_info["transformer_epoch"]

		# give a gap
		print()

		for epoch in range(start_epoch, EPOCHS):
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


				if (epoch + 1) % N_EPOCH_TO_EVALUATE == 0:
					# evaluate
					print("Evaluating...")
					results = master.evaluate(iter(val_datasets), max_seq_len)

					# save the results to file to be evaluated by COCO library
					with open("results/" + DATATYPE_VAL + "_captions_result.json", 'w') as outfile:
						json.dump(results, outfile)

					if len(results) != 0:
						# print metric evaluation
						cider = master.metric_eval("results/" + DATATYPE_VAL + "_captions_result.json")

						tf.summary.scalar('CIDEr', cider,
						                  step=epoch)  # REMEMBER: the epoch shown in the command line is epoch+1

						# based on the cider determine early stopping
						should_break = master.smart_ckpt_saver(epoch + 1,
						                                       cider)  # this will be better if we use validation
						if should_break == -1:
							start_epoch = epoch
							break
						elif should_break == 1:
							# store last epoch
							additional_info[key_epoch] = master.smart_ckpt_saver.max_acc_epoch
							store_additional_info(additional_info, ADDITIONAL_FILENAME)

			print()

		print('Saving Transformer weights for epoch {}'.format(master.smart_ckpt_saver.max_acc_epoch))
		master.ckpt.restore(master.ckpt_manager.latest_checkpoint)  # load checkpoint that was just trained to model
		master.transformer.save_weights(TRANSFORMER_WEIGHT_PATH)  # save the preprocessing weights