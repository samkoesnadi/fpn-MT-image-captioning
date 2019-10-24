"""
Print the evaluation result
"""

from common.common_imports import *
from common.common_definitions import *
from dataset import MetricEval

if __name__ == "__main__":
	metricEval = MetricEval(DATADIR, DATATYPE_VAL)

	imgIds = metricEval.coco.loadRes(RESULT_FILE).getImgIds()  # get all imgIds

	for i_imgID, imgID in enumerate(imgIds):
		print('---', i_imgID, imgID)
		# show result of the specific ID
		metricEval.print_result(imgID, RESULT_FILE)
		print()