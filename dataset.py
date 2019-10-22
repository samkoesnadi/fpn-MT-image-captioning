"""
Functions related to dataset handling

Main function: convert the dataset to TFRecord File
"""

from common.common_definitions import *
from common.common_imports import *
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pathlib import Path


# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def load_image(img_path, caption):
	# load image
	img = tf.io.read_file(img_path)
	img = tf.image.decode_jpeg(img, channels=3)
	img = tf.image.resize(img, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))
	img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

	return img, caption


def get_coco_images_dataset(dataDir, dataType, n_test=None):
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
	captions = ["<start> " + ann["caption"] + " <end>" for ann in anns]  # also put the start and end token
	imgIds = [ann["image_id"] for ann in anns]

	tokenizer_file = Path(TOKENIZER_FILENAME)
	if tokenizer_file.is_file():
		tokenizer = load_tokenizer_from_path(tokenizer_file)

		print("Tokenizer is loaded from", tokenizer_file)
	else:
		# preprocess captions into token
		tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=TOP_K,
		                                                  oov_token="unk",
		                                                  filters='!"#$%&()*+-/:;=?@[\]^_`{|}~ ')
		tokenizer.fit_on_texts(captions)

		# put padding in the dictionary
		tokenizer.word_index[''] = 0
		tokenizer.index_word[0] = ''

		store_tokenizer_to_path(tokenizer, TOKENIZER_FILENAME)

	# preprocess the captions to seperate ',' and '.' from words
	captions = [re.sub(r'([.,])', r" \1 ", caption) for caption in captions]

	# convert captions to sequences
	captions_token = tokenizer.texts_to_sequences(captions)

	set_len = math.ceil(len(captions_token) / BATCH_SIZE)
	max_seq_len = max(map(len, captions_token))

	# Pad each vector to the max_length of the captions
	# If you do not provide a max_length value, pad_sequences calculates it automatically
	captions_token = tf.keras.preprocessing.sequence.pad_sequences(captions_token, padding='post')

	imgs = coco.loadImgs(imgIds)
	img_paths = [os.path.join(dataDir, "images", dataType, img["file_name"]) for img in imgs]

	# Feel free to change batch_size according to your system configuration
	image_dataset = tf.data.Dataset.from_tensor_slices((img_paths, captions_token))
	image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	image_dataset = image_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
	image_dataset = image_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	return image_dataset, max_seq_len, set_len

def _tokenizer_from_json(json_string):
    """Parses a JSON tokenizer configuration file and returns a
    tokenizer instance.
    # Arguments
        json_string: JSON string encoding a tokenizer configuration.
    # Returns
        A Keras Tokenizer instance
    """
    tokenizer_config = json.loads(json_string)
    config = tokenizer_config.get('config')

    word_counts = json.loads(config.pop('word_counts'))
    word_docs = json.loads(config.pop('word_docs'))
    index_docs = json.loads(config.pop('index_docs'))
    # Integer indexing gets converted to strings with json.dumps()
    index_docs = {int(k): v for k, v in index_docs.items()}
    index_word = json.loads(config.pop('index_word'))
    index_word = {int(k): v for k, v in index_word.items()}
    word_index = json.loads(config.pop('word_index'))

    tokenizer = tf.keras.preprocessing.text.Tokenizer(**config)
    tokenizer.word_counts = word_counts
    tokenizer.word_docs = word_docs
    tokenizer.index_docs = index_docs
    tokenizer.word_index = word_index
    tokenizer.index_word = index_word

    return tokenizer

def load_tokenizer_from_path(path):
	"""

	:param path:
	:return:
	"""
	with open(path) as f:
		data = json.load(f)
		tokenizer = _tokenizer_from_json(data)

	return tokenizer

def store_tokenizer_to_path(tokenizer, path):
	"""

	:param tokenizer: Tokenizer object to be stored
	:param path: designated path for it
	:return:
	"""
	tokenizer_json = tokenizer.to_json()
	with open(path, 'w', encoding='utf-8') as f:
		f.write(json.dumps(tokenizer_json, ensure_ascii=False))


def get_coco_images_captions_generator(dataDir, dataType):
	"""
	Generator of images and five captions. This is used mainly for validation step.

	:param
	dataDir: directory of COCO
	dataType: which version do you want, e-g- val2017

	:return: tuple of the image and 5 references of ground truth caption
	"""

	# initialize COCO api for caption annotations
	annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)

	# initialize COCO api for instance annotations
	coco = COCO(annFile)

	# initialize tokenizer
	tokenizer_file = Path(TOKENIZER_FILENAME)
	if tokenizer_file.is_file():
		tokenizer = load_tokenizer_from_path(tokenizer_file)

		print("Tokenizer is loaded from", tokenizer_file)
	else:
		raise Exception("tokenizer is not yet created in", TOKENIZER_FILENAME)

	imgIds = coco.getImgIds()

	for imgId in imgIds:
		annIds = coco.getAnnIds(imgIds=imgId)

		anns = coco.loadAnns(annIds)
		captions = ["<start> " + ann["caption"] + " <end>" for ann in anns]  # also put the start and end token

		captions_token = tokenizer.texts_to_sequences(captions)

		imgs = coco.loadImgs(imgId)
		img_path = os.path.join(dataDir, "images", dataType, imgs[0]["file_name"])
		img, _ = load_image(img_path, None)

		yield img, captions_token

class COCO_Images_ImageID:
	def __init__(self, dataDir, dataType, n_val=None):
		"""
		Generator of images and five captions. This is used mainly for validation step.

		:param
		dataDir: directory of COCO
		dataType: which version do you want, e-g- val2017
		n_val: how many validation dataset do you want to have

		:return: (img, imgId)
		"""

		self.dataDir = dataDir
		self.dataType = dataType

		# initialize COCO api for caption annotations
		annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)

		# initialize COCO api for instance annotations
		self.coco = COCO(annFile)

		self.imgIds = self.coco.getImgIds()  # get n_val imgIds

		self.max_len = len(self.imgIds) if n_val is None else n_val
		self.imgIds = self.imgIds if n_val is None else self.imgIds[:n_val]

		self.iterIndex = 0

	def __iter__(self):
		self.iterIndex = 0
		return self

	def __next__(self):
		if self.iterIndex >= self.max_len:
			raise StopIteration

		imgId = self.imgIds[self.iterIndex]

		imgs = self.coco.loadImgs(imgId)
		img_path = os.path.join(self.dataDir, "images", self.dataType, imgs[0]["file_name"])
		img, _ = load_image(img_path, None)

		self.iterIndex += 1  # increment iterIndex

		return img, imgId


def store_additional_info(dict, filename):
	with open(filename, 'w') as outfile:
		json.dump(dict, outfile)

def load_additional_info(filename):
	try:
		with open(filename) as infile:
			data = json.load(infile)
	except:
			data = {}
	return data

class MetricEval:
	def __init__(self, dataDir, dataType):
		"""

		:param dataDir: of ground truth
		:param dataType:  of ground truth
		"""
		# initialize COCO api for caption annotations
		annFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)

		# initialize COCO api for instance annotations
		self.coco = COCO(annFile)

	def __call__(self, resFile):
		"""

		:param resFile:
		:return: CIDEr value. this can be changed with any metric you want actually. ("CIDEr","Bleu_4","Bleu_3","Bleu_2","Bleu_1","ROUGE_L","METEOR","SPICE")
		"""
		cocoRes = self.coco.loadRes(resFile)

		# create cocoEval object by taking coco and cocoRes
		cocoEval = COCOEvalCap(self.coco, cocoRes)

		# evaluate on a subset of images by setting
		# cocoEval.params['image_id'] = cocoRes.getImgIds()
		# please remove this line when evaluating the full validation set
		cocoEval.params['image_id'] = cocoRes.getImgIds()

		# evaluate results
		# SPICE will take a few minutes the first time, but speeds up due to caching
		cocoEval.evaluate()

		# return CIDEr value
		return cocoEval.eval["CIDEr"]

	def print_result(self, imgId, resFile, dataDir, dataType):
		"""

		:param resFile:
		:param imgId: destined imgId in the validation dataset that we are looking into
		:return:
		"""
		cocoRes = self.coco.loadRes(resFile)

		print('ground truth captions')
		annIds = self.coco.getAnnIds(imgIds=imgId)
		anns = self.coco.loadAnns(annIds)
		self.coco.showAnns(anns)

		print('\n')
		print('generated caption')
		annIds = cocoRes.getAnnIds(imgIds=imgId)
		anns = cocoRes.loadAnns(annIds)
		self.coco.showAnns(anns)

		img = self.coco.loadImgs(imgId)[0]
		I = io.imread('%s/images/%s/%s' % (dataDir, dataType, img['file_name']))
		plt.imshow(I)
		plt.axis('off')
		plt.show()




if __name__ == "__main__":
	dataDir = '../retinanet/datasets/coco'
	dataType = 'val2017'

	import time
	start = time.time()
	val_generator = get_coco_images_captions_generator(dataDir, dataType)

	# test the val generator
	print(next(val_generator))
	print(next(val_generator))

	# get dataset
	# image_dataset, length = get_coco_images_dataset(dataDir, dataType)
	print(time.time()-start)