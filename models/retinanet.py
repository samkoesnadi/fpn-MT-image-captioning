"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from common.common_imports import *
from common.common_definitions import *
import layers
from models import mobilenet
from .coattention import CoAttention_CNN


def default_classification_model(
    pyramid_feature_size=256,
    classification_feature_size=256,
    name='classification_submodel'
):
    """ Creates the default classification submodel.

    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.

    Returns
        A tf.keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    inputs  = tf.keras.layers.Input(shape=(None, None, pyramid_feature_size))

    outputs = inputs
    for i in range(2):
        outputs = tf.keras.layers.Conv2D(
            filters=classification_feature_size,
            activation=ACTIVATION,
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    """ Creates the default regression submodel.

    Args
        num_values              : Number of values to regress.
        num_anchors             : Number of anchors to regress for each feature level.
        pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name                    : The name of the submodel.

    Returns
        A tf.keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    inputs  = tf.keras.layers.Input(shape=(None, None, pyramid_feature_size))

    outputs = inputs
    for i in range(2):
        outputs = tf.keras.layers.Conv2D(
            filters=regression_feature_size,
            activation=ACTIVATION,
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    """ Creates the FPN layers on top of the backbone features.

    Args
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper
    P5_feature_size           = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5_feature_size, C4])
    P5           = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5', activation=ACTIVATION)(P5_feature_size)

    # add P5 elementwise to C4
    P4           = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = tf.keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4', activation=ACTIVATION)(P4)

    # add P4 elementwise to C3
    P3 = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = tf.keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3', activation="relu")(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on P5_feature_size"
    P6 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', activation=ACTIVATION)(P5_feature_size)
    P6 = tf.keras.layers.MaxPooling2D(name='P6')(P6)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', activation="relu")(P6)
    P7 = tf.keras.layers.MaxPooling2D(name='P7')(P7)

    return [P3, P4, P5, P6, P7]


def default_submodels():
    """ Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel and a classification submodel.

    Args

    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """
    return [
        ('regression', default_regression_model()),
        ('classification', default_classification_model())
    ]


def __build_model_pyramid(name, model, features):
    """ Applies a single submodel to each FPN level.

    Args
        name     : Name of the submodel.
        model    : The submodel to predict.
        features : The FPN features.

    Returns
        A tensor containing the response from the submodel on the FPN features.
    """
    return tf.keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    """ Applies all submodels to each FPN level.

    Args
        models   : List of submodels to run on each pyramid level (by default only regression, classifcation).
        features : The FPN features.

    Returns
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(anchor_parameters, features):
    """ Builds anchors for the shape of the features from FPN.

    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.

    Returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    """
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return tf.keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def retinanet(
    inputs,
    backbone_layers,
    create_pyramid_features = __create_pyramid_features,
    submodels               = None,
    name                    = 'retinanet'
):
    """ Construct a RetinaNet model on top of a backbone.

    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).

    Args
        inputs                  : tf.keras.layers.Input (or list of) for the input to the model.
        num_classes             : Number of classes to classify.
        num_anchors             : Number of base anchors.
        create_pyramid_features : Functor for creating pyramid features given the features C4, C5 from the backbone.
        submodels               : Submodels to run on each feature map (default is regression and classification submodels).
        name                    : Name of the model.

    Returns
        A tf.keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.

        The order of the outputs is as defined in submodels:
        ```
        [
            regression, classification, other[0], other[1], ...
        ]
        ```
    """

    if submodels is None:
        submodels = default_submodels()

    C3, C4, C5 = backbone_layers

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = create_pyramid_features(C3, C4, C5)

    # for all pyramid levels, run available submodels
    pyramids = __build_pyramid(submodels, features)

    return tf.keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


class FeatureExtractor(tf.keras.layers.Layer):
    """
    Feature extractor to be inputted into UMV Transformer
    """
    def __init__(self, retinanet_weight_path=None):
        super(FeatureExtractor, self).__init__()

        # declare retinanet model as backbone
        self.retinanet_model = mobilenet.mobilenet_retinanet(backbone='mobilenet224_1.0')

        # load weight to model
        if retinanet_weight_path is not None:
            self.retinanet_model.load_weights(retinanet_weight_path)

        regression_submodel = self.retinanet_model.get_layer("regression_submodel")
        classification_submodel = self.retinanet_model.get_layer("classification_submodel")

        regression_output = regression_submodel.layers[N_CONV_SUBMODULE].output
        classification_output = classification_submodel.layers[N_CONV_SUBMODULE].output

        # last layer of submodels
        regression = tf.keras.layers.Conv2D(1, 3, padding="same", activation="linear", kernel_initializer=KERNEL_INITIALIZER)(regression_output)
        classification = tf.keras.layers.Conv2D(NUM_OF_RETINANET_FILTERS, 3, padding="same", activation="linear", kernel_initializer=KERNEL_INITIALIZER)(classification_output)

        # co-attention, CNN, downsample, CNN
        coatt_output, coatt_output_att_weights = CoAttention_CNN()(regression, classification)
        out = tf.keras.layers.Conv2D(NUM_OF_RETINANET_FILTERS, 3, padding="same", activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER)(coatt_output)
        out = tf.keras.layers.MaxPooling2D()(out)

        # linear layer to d_model
        out = tf.keras.layers.Dense(d_model, activation=ACTIVATION, kernel_initializer=KERNEL_INITIALIZER)(out)

        # remove last layer in the models
        submodel = tf.keras.Model(inputs=[regression_submodel.inputs, classification_submodel.inputs], outputs=[out, coatt_output_att_weights])  # output the coatt weight as well

        # compute the features
        features = [self.retinanet_model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]
        extracted_features = [submodel([feature, feature]) for feature in features]

        # build model out of extracted_features
        self.model = tf.keras.Model(self.retinanet_model.inputs, extracted_features)

    def call(self, inp):
            return self.model(inp)