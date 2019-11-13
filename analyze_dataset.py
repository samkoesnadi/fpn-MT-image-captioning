"""
Analyze dataset.
input's image cosine similarity, output's length distribution, vocabulary's distribution, cosine similarity, tf-idf in the training dataset
"""

from dataset import *
import pandas as pd

flatten = lambda l: [item for sublist in l for item in sublist]

def analyze_dataset(dataDir, dataType, n_test=None):
	"""
	Get only images as datasets. This function is designed for autoencoder in image_feature_extract.py

	:param
	dataDir: directory of COCO
	dataType: which version do you want, e-g- val2017
	n_test: how many dataset do you want to have

	:return: tf.data.Dataset of (image, caption), max_sequence_len
	"""

	# initialize COCO api for caption annotations
	annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)

	# initialize COCO api for instance annotations
	coco = COCO(annFile)

	annIds = coco.getAnnIds()[:n_test] if n_test is not None else coco.getAnnIds()

	anns = coco.loadAnns(annIds)
	anns = list(filter(lambda ann: ann["caption"] != ' ', anns))  # filter out empty data caption
	captions = get_captions_from_anns(anns)
	imgIds = [ann["image_id"] for ann in anns]

	# preprocess captions into token
	tokenizer = TokenTextEncoder_alphanumeric(captions, oov_token=" ", lowercase=True)

	# convert captions to sequences
	captions_token = tokenizer_encode(tokenizer, captions)

	# # Pad each vector to the max_length of the captions
	# # If you do not provide a max_length value, pad_sequences calculates it automatically
	# captions_token = tf.keras.preprocessing.sequence.pad_sequences(captions_token, padding='post')

	imgs = coco.loadImgs(imgIds)
	img_paths = [os.path.join(dataDir, "images", dataType, img["file_name"]) for img in imgs]

	# sort captions based on length and also the img with it
	data_raw = list(zip(img_paths, captions_token))
	data_raw.sort(key=lambda x: len(x[1]))

	flatten_words_count = flatten(captions_token)
	sequences_length = map(len, captions_token)
	pd_sequences_length = pd.Series(sequences_length)
	pd_words_count = pd.Series(flatten_words_count)

	# run counts function
	df_seq_len = pd_sequences_length.value_counts().sort_index()
	df_words_count = pd_words_count.value_counts().sort_index()

	ax = df_words_count.plot.bar(rot=0)
	ax.set_xlabel('seq len', fontsize=1)
	# plt.show()

	print(pd_words_count.describe())
	print("Vocab size:", tokenizer.vocab_size)


if __name__ == "__main__":
	analyze_dataset(DATADIR, DATATYPE_TRAIN)