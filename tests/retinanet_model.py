from common.common_definitions import *
from common.common_imports import *
from models import mobilenet

if __name__ == "__main__":
	input = tf.keras.layers.Input((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 3))
	model = mobilenet.mobilenet_retinanet(NUM_OF_CLASSES, backbone='mobilenet224_1.0', inputs=input)

	# load weight to model
	model.load_weights('../../retinanet/snapshots/mobilenet224_1.0_coco_01.h5')

	regression_submodel = model.get_layer("regression_submodel")
	classification_submodel = model.get_layer("classification_submodel")

	# remove last layer in the models
	regression_submodel = tf.keras.Model(regression_submodel.inputs, regression_submodel.layers[N_CONV_SUBMODULE].output)
	classification_submodel = tf.keras.Model(classification_submodel.inputs, classification_submodel.layers[N_CONV_SUBMODULE].output)

	# compute the anchors
	features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]

	regression_features = [regression_submodel(feature) for feature in features]
	classification_features = [classification_submodel(feature) for feature in features]

	print(regression_features)

	# print(model.summary())
