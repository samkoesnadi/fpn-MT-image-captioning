from common.common_imports import *
from common.common_definitions import *
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from tensorflow_probability.python.distributions import Multinomial

# use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
	print('TF uses GPU')
else:
	print("TF does not use GPU")

if __name__ == "__main__":
	cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction="none")
	loss = cce(
		np.array([[0, 0, 1], [1,0,0], [0,0,1]]),
		np.array([[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]]))
	print('Loss: ', loss.numpy())  # Loss: 0.3239