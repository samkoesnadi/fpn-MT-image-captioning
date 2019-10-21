"""
Evaluation functions, i.e. BLEU-n, SPICE, Rouge-L, METEOR, CIDEr-D

''' Discontinued ***
"""

import nltk


def score_bleu(reference, candidate, n=4):
	"""

	:param reference: e.g. [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
	:param candidate:
	:param n: the n-gram
	:return:
	"""

	# n has to be above 0
	if n < 1:
		return 0

	weights = [1/n] * n  # weights for the nltk function
	chencherry = nltk.translate.bleu_score.SmoothingFunction()
	try:
		score = nltk.translate.bleu_score.sentence_bleu(reference, candidate, weights=weights, smoothing_function=chencherry.method1)
	except:
		score = 0.
	return score

def score_SPICE(reference, candidate):
	pass

def score_ROUGEL(reference, candidate):
	pass

def score_METEOR(reference, candidate):
	pass

def score_CIDErD(reference, candidate):
	pass

if __name__ == "__main__":
	reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
	candidate = ['this', 'is', 'a', 'test']

	# test bleu
	score = score_bleu(reference, candidate)
	print("bleu = ", score)