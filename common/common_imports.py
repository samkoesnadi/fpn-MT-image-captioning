import tensorflow as tf
import numpy as np
import os
import skimage.io as io
import matplotlib.pyplot as plt
import json
from glob import glob
from PIL import Image
import pickle
import re
import time
import math
import random

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt
