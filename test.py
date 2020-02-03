# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 22:35:26 2020

@author: Viraj
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  6 22:19:36 2019

@author: Viraj
"""

# USAGE
#python predict.py --image images/dog.jpg --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --width 32 --height 32 --flatten 1
#python predict.py --image images/dog.jpg --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --width 64 --height 64

# import the necessary packages
from keras.models import load_model
import argparse
import pickle
import cv2
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import applications
from keras.models import Sequential, Model
from imutils import paths
import json




args = {'image' : 'testing/test1.jpg', 'model': 'output/neural_network.h5', 'label-bin': 'output/labels.pickle', 'width' : 20, 'height':20,'flatten':1}



# load the input image and resize it to the target spatial dimensions
img = cv2.imread(args["image"])
output = img.copy()



 # Resize
height, width, _ = img.shape
new_height = height * 64 // min(img.shape[:2])
new_width = width * 64 // min(img.shape[:2])
img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

# Crop
height, width, _ = img.shape
startx = width//2 - (64//2)
starty = height//2 - (64//2)
img = img[starty:starty+64,startx:startx+64]
assert img.shape[0] == 64 and img.shape[1] == 64, (img.shape, height, width)


data=[]
data.append(img)

# scale the raw pixel intensities to the range [0, 1]
img = np.array(data, dtype="float") / 255.0



# load the model and label binarizer
print("[INFO] loading network and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label-bin"], "rb").read())

# make a prediction on the image
preds = model.predict(img)
print(preds)

# find the class label index with the largest corresponding
# probability
i = preds.argmax(axis=1)[0]
#label = lb.classes_[i]
#target_names = ['Humans', 'Animals']
label = lb.classes_[i]

# draw the class label + probability on the output image
text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(0, 0, 255), 2)

# show the output image
cv2.imshow("Image", output)
cv2.waitKey(0)