from common.common_imports import *
from common.common_definitions import *
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
	print('TF uses GPU')
else:
	print("TF does not use GPU")

if __name__ == "__main__":
	cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
	loss = cce(
		[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
		[[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]])
	print('Loss: ', loss.numpy())  # Loss: 0.0945

	softmaxed_target = tf.nn.softmax([[.9, .05, .05], [.05, .89, .06], [.05, .01, .94]])
	cce = tf.keras.losses.CategoricalCrossentropy()
	loss = cce(
		[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
		softmaxed_target)
	print('Loss: ', loss.numpy())  # Loss: 0.0945