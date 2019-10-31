from common.common_imports import *
from common.common_definitions import *
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

if __name__ == "__main__":
	cider_score_eval = Cider()
	# gts = {1078: [{'image_id': 1078, 'caption': 'todaytodaytodaytodaytodaytodaytodaytodaytodaytodaytodaytodaytodaytodaytodaytodaytoday', 'id': 1}]}
	#
	#
	#
	# print('tokenization...')
	# tokenizer = PTBTokenizer()
	# gts = tokenizer.tokenize(gts)
	# res = tokenizer.tokenize(gts)
	gts = {1078: ['fdsafdsafdsafdsafd fdsa fdsa asdf fds']}
	res = {1078: ['fdsafdsafdsafdsafdsafdsafdsafdsafdsa']}
	print(cider_score_eval.compute_score(gts, res))