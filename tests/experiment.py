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
	p = [[[.1, .2, .7], [.3, .3, .4]], [[.1, .2, .7], [.3, .3, .4]]]  # Shape [2, 3]
	dist = Multinomial(total_count=1, probs=p)

	counts = [[2., 1, 1], [3, 1, 1]]
	print(dist.prob(counts))  # Shape [2]

	mask = (dist.sample(1)[0]) # Shape [5, 2, 3]
	print(mask)

	b = p * mask
	c = tf.math.reduce_max(b, axis=-1)
	# a = tf.reshape(tf.boolean_mask(p, mask), tf.shape(mask)[0:-1])
	print(b,c)

	print(tf.reduce_mean([0., .5, 0., 0.]))
