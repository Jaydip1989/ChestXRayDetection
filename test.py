import numpy as np 
import keras
import tensorflow
import math
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import Adam
from tensorflow.python.framework import ops
ops.reset_default_graph()
import PIL
from PIL import Image

json_file = open('./model_adam_1.json')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.compile(loss = "categorical_crossentropy",
					optimizer = Adam(lr=0.0001),
					metrics=['accuracy'])





IMG = Image.open('/Users/dipit/chest_xray/val/test8.jpeg')
print(type(IMG))
IMG = IMG.resize((64, 64))
IMG = np.array(IMG)
IMG = np.true_divide(IMG, 255)
IMG = IMG.reshape(1,64,64,1)


predictions = loaded_model.predict(IMG)
predictions_c = loaded_model.predict_classes(IMG)
print(predictions, predictions_c)

classes = {'TRAIN':['NORMAL', 'PNEUMONIA'],
			'VALIDATION':['NORMAL', 'PNEUMONIA']
			}
predicted_class = classes['TRAIN'][predictions_c[0]]
print("We think that is {}.".format(predicted_class.lower()))