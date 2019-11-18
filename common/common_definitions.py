import logging
import tensorflow as tf
import os


IS_TRAINING = True

USE_GPU = True
LOG_ATTENTION = False  # to output the attention layers as well

LOGGING_LEVEL = logging.DEBUG

ACTIVATION = tf.nn.relu
KERNEL_INITIALIZER = tf.keras.initializers.glorot_uniform()
KERNEL_REGULARIZER_LAMBDA = 1e-5

### Set default parameters for all model
IMAGE_INPUT_SIZE = 512  # this to fit default criteria from MobileNetV2-retinanet
BATCH_SIZE = 10  # for the SCST
XE_BATCH_SIZE = 12
BUFFER_SIZE = 2000  # this is important for shuffling
EPOCHS = 50
XE_EPOCHS = 30  # the amount of epoch cross entropy will go through.
BEAM_SEARCH_N = 4
N_VAL_DATASET = 50  # the number of dataset to be validated
N_TRAIN_DATASET = None  # the number of dataset to be trained
N_EPOCH_TO_EVALUATE = 1  # rythm of the epoch to evaluate and save checkpoint
MAX_EVAL_TIME = 180 # maximum evaluation time in seconds
AMOUNT_OF_VALIDATION = 100  # used for convert_dataset
DROPOUT_RATE = 0.3

# dataset preprocessing parameter
P_AUGMENTATION = 0.8
END_TOKEN = "<EOS>"
FINDINGS_TOKEN = "<FINDINGS>"
IMPRESSION_TOKEN = "<IMPRESSION>"
MAX_WORDS_LEN = 60
TOP_K = 1000  # this is for tokenizer

# MIN_EPOCH_TO_BREAK = EPOCHS // 2
MIN_EPOCH_TO_EVAL = 10
MIN_EPOCH_TO_SAVE_CKPT = 5
MAX_CKPT_TO_KEEP = 10
# GAP_OF_DEAD_EPOCH = 25  # gap before it is going to kill the no more training network
WARM_UP_STEPS = 4000  # for scheduler
CKPT_INDEX_RESTORE = -1  # -1 for the last one
START_EPOCH = None  # set it to None, otherwise it overrides the start epoch from additional file

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
TOKENIZER_FILENAME = "datasets/_tokenizer"
ADDITIONAL_FILENAME = "datasets/_additional_extractor.json"
RETINANET_WEIGHT_PATH = None  # autoencoder trained on pix2code datasets
TRANSFORMER_WEIGHT_PATH = "model_weights/multimodal_transformer.h5"  # transformer trai
TRANSFORMER_CHECKPOINT_PATH = "./checkpoints/train/multimodal_transformer"
RESULT_FILE = "results/" + DATATYPE_VAL + "_captions_result.json"
PLOT_RESULT_FOLDER = "results/plots/"
LOGGING_FILE_PATH = "results/_logging_multimodal_transformer.txt"

### Set Hyperparameters for Transformer
num_layers = 6
d_model = 512
dff = 2048
num_heads = 8


### Set parameter for RetinaNet
NUM_OF_RETINANET_FILTERS = 256
NUM_OF_PYRAMIDS = 4
N_CONV_SUBMODULE = 2  # how many times the intermediate CNNs is repeated in the submodules

# MT-UMV-Encoder
BASELINE_INDEX = 1  # index of the baseline in the pyramids array. range is 0 to NUM_OF_PYRAMIDS-1  (the less the bigger)

# SCST's parameter
XE_LEARNING_EPSILON = 1e-7
SCST_LEARNING_EPSILON = 1e-6
MAX_SCST_LEARNING_RATE = 1e-6
MIN_SCST_LEARNING_RATE = 5e-7
REWARD_DISCOUNT_FACTOR = 100.  # as in the SCST paper, the CIDEr is always in range above 100, while what I have is always in range 1
MAX_TEMPERATURE = 1.


# testing
MAX_SEQ_LEN_ATT_PLOT = 10

logging.basicConfig(level=LOGGING_LEVEL)

if not USE_GPU:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

	if tf.test.gpu_device_name():
		print('TF uses GPU')
	else:
		print("TF does not use GPU")

mirrored_strategy = tf.distribute.MirroredStrategy()  # strategy for distributed training

logging.info ('Number of GPU devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

# multiply the batch size according to the num of replicas
BATCH_SIZE *= mirrored_strategy.num_replicas_in_sync
XE_BATCH_SIZE *= mirrored_strategy.num_replicas_in_sync
