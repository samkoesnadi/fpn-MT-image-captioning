"""
test model with input file path, store the attention layer to file
"""

from common.common_definitions import *
from common.common_imports import *
from utils.pipeline import *
from utils.utils import *
from dataset import *
import shutil

if __name__ == "__main__":
	IMAGE_FILE_PATH = "CXR86_IM-2380-1001.png"
	imgId = 0

	max_seq_len = load_additional_info(ADDITIONAL_FILENAME)["max_seq_len"]
	master = Pipeline(TOKENIZER_FILENAME, TRANSFORMER_CHECKPOINT_PATH, max_seq_len)  # master pipeline

	# evaluate
	print("Evaluating...")

	# load image from file
	img = load_image(IMAGE_FILE_PATH)
	result, attention_weights, coatt_weights = master.evaluate_img(img, max_seq_len)

	results = [{
		"image_id": imgId,
		"caption": result
	}]

	# save the results to file to be evaluated by COCO library
	test_file_name = IMAGE_FILE_PATH.split('.')[0]
	with open("results/" + test_file_name + "_captions_result.json", 'w') as outfile:
		json.dump(results, outfile)

	if LOG_ATTENTION:
		plot_result_folder_test = os.path.join(PLOT_RESULT_FOLDER, test_file_name + '/')
		# plot the attention weights and store them
		try:
			shutil.rmtree(plot_result_folder_test, ignore_errors=True)
			os.makedirs(plot_result_folder_test + "coatts")
			os.makedirs(plot_result_folder_test + "dec_atts")
		except:
			pass

		# preprocess caption to be the title of the plot
		caption = results[0]["caption"] + " <end>"
		caption_tokenized = caption.split()

		# plot the coattention weights as saliency map w/ respect to image
		logging.info("Plot co-attention weights and store to files")
		results = plot_coatt(img, coatt_weights)

		for i_result, (saliency_map, original_size) in enumerate(results):
			plt.imshow(saliency_map, interpolation='nearest')
			plt.title("feature map size: " + str(original_size) + "x" + str(original_size))
			plt.axis('off')
			plt.savefig(plot_result_folder_test + "coatts/coatts_" + str(i_result) + ".png", bbox_inches='tight')
			plt.close()

		# plot the attention weights
		logging.info("Plot attention weights and store to files")
		for i_saliency, saliency_map in enumerate(plot_att(img, attention_weights)):
			try:
				os.makedirs(plot_result_folder_test + "dec_atts/layer" + str(i_saliency))
			except:
				pass

			for i_word, word in enumerate(saliency_map):
				title = caption_tokenized[i_word]

				fig, axs = plt.subplots(math.ceil(math.sqrt(num_heads)), math.ceil(math.sqrt(num_heads)), constrained_layout=True)
				axs = axs.flatten()  # flatten it so we do not need any modulus later
				fig.suptitle(title, fontsize=16)

				for i_head, head in enumerate(word):
					head = (head - tf.math.reduce_min(head)) / (
							tf.math.reduce_max(head) - tf.math.reduce_min(head))  # change range
					axs[i_head].imshow(head)
					axs[i_head].set_title('Head {}'.format(i_head + 1))

				[axs[i].axis('off') for i in range(axs.size)]  # don't show axis

				# save the figure
				fig.savefig(plot_result_folder_test + "dec_atts/layer" + str(i_saliency) + '/' + str(i_word) + ".png", bbox_inches='tight')
				plt.close()