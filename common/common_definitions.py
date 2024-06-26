import logging
import tensorflow as tf
import os


IS_TRAINING = True

USE_GPU = True

LOGGING_LEVEL = logging.DEBUG

TOP_K = 10000  # this is for tokenizer

ACTIVATION = tf.nn.leaky_relu
KERNEL_INITIALIZER = tf.keras.initializers.he_normal()

### Set default parameters for all model
IMAGE_INPUT_SIZE = 512  # this to fit default criteria from MobileNetV2-retinanet
BATCH_SIZE = 10
BUFFER_SIZE = 1000  # this is important for shuffling
EPOCHS = 100
BEAM_SEARCH_N = 4
N_VAL_DATASET = 50  # the number of dataset to be validated
N_TRAIN_DATASET = None  # the number of dataset to be trained
N_EPOCH_TO_EVALUATE = 1  # rythm of the epoch to evaluate and save checkpoint
AMOUNT_OF_VALIDATION = 100  # used for convert_dataset
DROPOUT_RATE = 0.1

MIN_EPOCH_TO_BREAK = EPOCHS // 2
GAP_OF_DEAD_EPOCH = 25  # gap before it is going to kill the no more training network
# INIT_LEARNING_RATE = 1e-4
WARM_UP_STEPS = 4000  # for scheduler

# Dataset Directory

# # coco
# DATADIR = '../retinanet/datasets/coco'
# DATATYPE_VAL = 'val2017'
# DATATYPE_TRAIN = 'train2017'

# # xray
DATADIR = 'datasets/iuxray'
DATATYPE_VAL = 'val2017'
DATATYPE_TRAIN = 'train2017'


# filenames
TOKENIZER_FILENAME = "datasets/_tokenizer.json"
ADDITIONAL_FILENAME = "datasets/_additional_extractor.json"
RETINANET_WEIGHT_PATH = "model_weights/mobilenet224_1.0_coco.h5"  # autoencoder trained on pix2code datasets
TRANSFORMER_WEIGHT_PATH = "model_weights/multimodal_transformer.h5"  # transformer trai
TRANSFORMER_CHECKPOINT_PATH = "./checkpoints/train/multimodal_transformer"
RESULT_FILE = "results/" + DATATYPE_VAL + "_captions_result.json"

### Set Hyperparameters for Transformer
num_layers = 6
d_model = 512
dff = 2048
num_heads = 8


### Set parameter for RetinaNet
NUM_OF_CLASSES = 80
NUM_OF_RETINANET_FILTERS = 256
NUM_OF_ANCHORS = 9
NUM_OF_PYRAMIDS = 5
N_CONV_SUBMODULE = 2  # how many times the intermediate CNNs is repeated in the submodules

# MT-UMV-Encoder
BASELINE_INDEX = 3  # index of the baseline in the pyramids array. range is 0 to NUM_OF_PYRAMIDS-1


logging.basicConfig(level=LOGGING_LEVEL)

if not USE_GPU:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

	if tf.test.gpu_device_name():
		print('TF uses GPU')
	else:
		print("TF does not use GPU")