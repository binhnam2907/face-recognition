import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model

def model_resnet():
    resnet = ResNet50(
    input_shape = (256,256,3),
    weights = 'imagenet', 
    include_top = False)
    model = Model(inputs=resnet.inputs, outputs=resnet.layers[-1].output)

    return model

def return_feature(image1):
    image1 = cv2.resize(image1, (256,256))
    image1 = np.expand_dims(image1, axis=0)
    resnet = model_resnet()
    model = Model(inputs=resnet.inputs, outputs=resnet.layers[-1].output)
    feature = model.predict(image1)
    return feature