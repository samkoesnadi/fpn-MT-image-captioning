"""
Load train and validation datasets for hierachical transformer
"""

from dataset import *
from common.common_definitions import *
from common.common_imports import *
import itertools

def group_list_by_delimiter(lst, delimiter):
	return [list(y) for x, y in itertools.groupby(lst, lambda z: z == delimiter) if not x]

def _pad_list(lst, N, pad=0):
	lst += [pad] * (N - len(lst))
	return lst

def H_get_coco_images_dataset(dataDir, dataType, n_test=None):
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
	captions = [ann["caption"] + " <end>" for ann in anns]  # also put the start and end token
	imgIds = [ann["image_id"] for ann in anns]

	tokenizer_file = Path(TOKENIZER_FILENAME)
	if tokenizer_file.is_file():
		tokenizer = load_tokenizer_from_path(tokenizer_file)

		print("Tokenizer is loaded from", tokenizer_file)
	else:
		# preprocess captions into token
		tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=TOP_K,
		                                                  oov_token="<unk>",
		                                                  filters='!"#$%&()*+-/:;=?@[\]^_`{|}~ ')
		tokenizer.fit_on_texts(captions)

		tokenizer.word_index['<pad>'] = 0
		tokenizer.index_word[0] = '<pad>'

		store_tokenizer_to_path(tokenizer, TOKENIZER_FILENAME)

	# preprocess the captions to seperate ',' and '.' from words
	captions = [re.sub(r'([.,])', r" \1 ", caption) for caption in captions]

	# convert captions to sequences
	captions_token = tokenizer.texts_to_sequences(captions)

	dot_token = tokenizer.word_index["."]
	start_token = tokenizer.word_index["<start>"]
	end_token = tokenizer.word_index["<end>"]
	groups_sentences = [group_list_by_delimiter(caption_token, dot_token) for caption_token in captions_token]  # sentences groupped by delimiter

	set_len = math.ceil(len(captions_token) / BATCH_SIZE)  # total steps in one epoch

	# max_seq_len = max(map(len, captions_token))
	max_sentence_len = max(map(len, groups_sentences))
	max_words_len = max([max(map(len, dataset)) for dataset in groups_sentences])

	def process_sentence(sentence):
		if sentence[0] != end_token:
			return _pad_list([start_token] + sentence + [end_token], max_words_len, 0)
		else:
			return _pad_list(sentence, max_words_len, 0)

	groups_sentences = [_pad_list(list(map(process_sentence, dataset)), max_sentence_len, [0]) for dataset in groups_sentences]

	imgs = coco.loadImgs(imgIds)
	img_paths = [os.path.join(dataDir, "images", dataType, img["file_name"]) for img in imgs]

	# Feel free to change batch_size according to your system configuration
	image_dataset = tf.data.Dataset.from_tensor_slices((img_paths, groups_sentences))
	image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	image_dataset = image_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
	image_dataset = image_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	return image_dataset, max_seq_len, set_len

if __name__ == "__main__":
	train_datasets, max_seq_len, train_set_len = H_get_coco_images_dataset(DATADIR, DATATYPE_TRAIN, N_TRAIN_DATASET)