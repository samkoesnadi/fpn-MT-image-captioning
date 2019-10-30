"""
Functions related to dataset handling

Main function: convert the dataset to TFRecord File
"""

from common.common_definitions import *
from common.common_imports import *
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pathlib import Path
from random import shuffle
import albumentations

# apply augmentation
def _aug(p=0.5):
	return albumentations.OneOf([
		albumentations.augmentations.transforms.ElasticTransform(alpha=88, sigma=10, p=0.5),
		albumentations.augmentations.transforms.ShiftScaleRotate(scale_limit=(-.1,0), border_mode=0, rotate_limit=5, p=0.5),
		albumentations.augmentations.transforms.ShiftScaleRotate(scale_limit=(-.2,.2), rotate_limit=10, p=0.5)
	], p=p)

# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def load_image(img_path, augmentation=None):
	# load image
	img = tf.io.read_file(img_path)
	img = tf.image.decode_jpeg(img, channels=3)
	img = tf.image.resize(img, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE))

	if augmentation is not None:
		img_dtype = img.dtype
		img_shape = tf.shape(img)
		image = tf.numpy_function(lambda x: augmentation(image=x / 255.)["image"],
		                          [img],
		                          img_dtype)  # augment the image
		img = tf.reshape(image, shape=img_shape)

	return img

def tokenizer_encode(tokenizer, captions):
	captions_token = [[tokenizer.vocab_size] + tokenizer.encode(caption) + [tokenizer.vocab_size + 1] for caption in captions]  # vocab_size is <start> and vocab_size+1 is <end>

	return captions_token

def load_image_and_preprocess(img_path, caption, augmentation=None):
	# load image
	img = load_image(img_path, augmentation)
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
	anns = list(filter(lambda ann: ann["caption"] != ' ', anns))  # filter out empty data caption
	captions = [ann["caption"] for ann in anns]  # also put the start and end token
	imgIds = [ann["image_id"] for ann in anns]

	tokenizer_file = Path(TOKENIZER_FILENAME)
	if tokenizer_file.is_file():
		tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(TOKENIZER_FILENAME)

		print("Tokenizer is loaded from", tokenizer_file)
	else:
		# preprocess captions into token
		tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(captions, target_vocab_size=TOP_K)

		# store the tokenizer
		tokenizer.save_to_file(TOKENIZER_FILENAME)

	# preprocess the captions to seperate ',' and '.' from words
	captions = [re.sub(r'([.,])', r" \1 ", caption) for caption in captions]

	# convert captions to sequences
	captions_token = tokenizer_encode(tokenizer, captions)

	captions_token_len = len(captions_token)

	# set buffer size if BUFFER_SIZE = None
	buffer_size = captions_token_len if BUFFER_SIZE is None else BUFFER_SIZE

	set_len = math.ceil(captions_token_len / BATCH_SIZE)
	max_seq_len = max(map(len, captions_token))

	# # Pad each vector to the max_length of the captions
	# # If you do not provide a max_length value, pad_sequences calculates it automatically
	# captions_token = tf.keras.preprocessing.sequence.pad_sequences(captions_token, padding='post')

	imgs = coco.loadImgs(imgIds)
	img_paths = [os.path.join(dataDir, "images", dataType, img["file_name"]) for img in imgs]

	# generator for input to dataset
	def dataset_generator():
		for (img_path, caption_token) in zip(img_paths, captions_token):
			yield (img_path, caption_token)

	# Feel free to change batch_size according to your system configuration
	image_dataset = tf.data.Dataset.from_generator(dataset_generator, output_types=(tf.string, tf.float32), output_shapes=(tf.TensorShape([]), tf.TensorShape([None])))

	# set augmentation
	augmentation = _aug(p=P_AUGMENTATION)
	image_dataset = image_dataset.map(lambda img, caption: load_image_and_preprocess(img, caption, augmentation), num_parallel_calls=tf.data.experimental.AUTOTUNE)  # load the image
	image_dataset = image_dataset.shuffle(buffer_size).padded_batch(BATCH_SIZE, padded_shapes=([None, None, None], [-1]), drop_remainder=True)  # shuffle and batch with length of padding according to the the batch
	image_dataset = image_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	return image_dataset, max_seq_len, set_len

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
		tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(TOKENIZER_FILENAME)

		print("Tokenizer is loaded from", tokenizer_file)
	else:
		raise Exception("tokenizer is not yet created in", TOKENIZER_FILENAME)

	imgIds = coco.getImgIds()

	for imgId in imgIds:
		annIds = coco.getAnnIds(imgIds=imgId)

		anns = coco.loadAnns(annIds)
		anns = list(filter(lambda ann: ann["caption"] != ' ', anns))  # filter out empty data caption
		captions = [ann["caption"] for ann in anns]  # also put the start and end token

		captions_token = tokenizer_encode(tokenizer, captions)

		imgs = coco.loadImgs(imgId)
		img_path = os.path.join(dataDir, "images", dataType, imgs[0]["file_name"])
		img = load_image(img_path)

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

		# filter out all empty datasets
		annIds = self.coco.getAnnIds()
		anns = self.coco.loadAnns(annIds)
		anns = filter(lambda ann: ann["caption"] != ' ', anns)  # filter out empty data caption

		self.imgIds = list(map(lambda ann: ann["image_id"], anns))  # get n_val imgIds

		# randomize the imgIds, so the evaluation will be fair
		shuffle(self.imgIds)

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
		img = load_image(img_path)

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

		self.dataDir = dataDir
		self.dataType = dataType

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

	def print_result(self, imgId, resFile):
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
		I = io.imread('%s/images/%s/%s' % (self.dataDir, self.dataType, img['file_name']))
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