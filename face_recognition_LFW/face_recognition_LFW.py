'''
Face Recognition (LFW Dataset from Kaggle) Dataset provided by UMass

Goals: Train a CNN that is able to accurately identify faces to names. 

Each image is available as "lfw/name/name_xxxx.jpg" where "xxxx" is the image number padded to four characters with leading zeros
Example: "lfw/George_W_Bush/George_W_Bush_0010.jpg"

Total 13233 images of 5749 people in the database. 

Each image is a 250x250 jpg


'''

import tensorflow as tf
import numpy as np
from PIL import Image

