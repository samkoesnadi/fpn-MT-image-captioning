"""
test model with input file path
"""

from common.common_definitions import *
from common.common_imports import *
from utils.pipeline import *
from dataset import *

if __name__ == "__main__":
	IMAGE_FILE_PATH = "test_1.jpeg"

	max_seq_len = load_additional_info(ADDITIONAL_FILENAME)["max_seq_len"]
	master = Pipeline(TOKENIZER_FILENAME, TRANSFORMER_CHECKPOINT_PATH, max_seq_len)  # master pipeline

	# evaluate
	print("Evaluating...")

	# load image from file
	img, _ = load_image(IMAGE_FILE_PATH, None)
	results = master.evaluate_img(img, max_seq_len)

	# save the results to file to be evaluated by COCO library
	with open("results/" + IMAGE_FILE_PATH.split('.')[0] + "_captions_result.json", 'w') as outfile:
		json.dump(results, outfile)