"""
Copyright 2017-2018 lvaleriu (https://github.com/lvaleriu/)

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

from . import retinanet
from . import Backbone


class MobileNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    allowed_backbones = ['mobilenet128', 'mobilenet160', 'mobilenet192', 'mobilenet224']

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return mobilenet_retinanet(*args, backbone=self.backbone, **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        backbone = self.backbone.split('_')[0]

        if backbone not in MobileNetBackbone.allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, MobileNetBackbone.allowed_backbones))


def mobilenet_retinanet(backbone='mobilenet224_1.0', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a mobilenet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('mobilenet128', 'mobilenet160', 'mobilenet192', 'mobilenet224')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a MobileNet backbone.
    """
    alpha = float(backbone.split('_')[1])

    # choose default input
    if inputs is None:
        inputs = tf.keras.layers.Input((None, None, 3))

    backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(input_tensor=inputs, alpha=alpha, include_top=False, pooling=None, weights="imagenet")

    # create the full model
    # layer_names = ['block_5_add', 'block_12_add', 'out_relu']  # this is C3, C4, C5
    layer_names = ['block_12_add', 'out_relu']
    layer_outputs = [backbone.get_layer(name).output for name in layer_names]
    backbone = tf.keras.models.Model(inputs=inputs, outputs=layer_outputs, name=backbone.name)

    # invoke modifier if given
    if modifier:
        backbone = modifier(backbone)

    return retinanet.retinanet(inputs=inputs, backbone_layers=backbone.outputs, **kwargs)
