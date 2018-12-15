from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
import coremltools
import pickle
import urllib

urllib.request.urlretrieve("https://s3.amazonaws.com/csye7374/config.json", filename= 'config.json')

urllib.request.urlretrieve("https://s3.amazonaws.com/csye7374/cifar10.h5", filename= 'cifar10.h5')

json_file = open('config.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

keras_model = model_from_json(loaded_model_json)

keras_model.load_weights("cifar10.h5")

coreml_model = coremltools.converters.keras.convert(keras_model,
	input_names="image",
	image_input_names="image",
	image_scale=1/255.0,
	class_labels=class_labels,
	is_bgr=True)

coreml_model.save('cifar10CM.mlmodel')