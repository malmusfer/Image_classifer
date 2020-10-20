#import import libraries
from PIL import Image 
import numpy as np 
import argparse 
import tensorflow as tf
import tensorflow_hub as hub
import json
import warnings
warnings.filterwarnings('ignore')
import logging
import matplotlib.pyplot as plt


logger = tf.get_logger()
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser ()
parser.add_argument ('--image_path', type = str)
parser.add_argument('--model', type=str)
parser.add_argument ('--topk', type = int)
parser.add_argument ('--clabel_map', type = str)
commands = parser.parse_args()
print(commands)

# function to load class names
def load_class_names(json_path):
    with open(json_path) as n:
        class_n = json.load(n)

    the_name = dict()
    for key in class_n.keys():
        the_name[str(int(key))] = class_n[key]

    return the_name

# function to load the model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    return model

# function to process the image
def process_image(img):
    process_image = np.squeeze(img)
    process_image = tf.image.resize(process_image, (224, 224))/255.0

    return process_image


image_path, model, K, label_map = commands.image_path, commands.model, commands.topk, commands.clabel_map

# prediction function
def predict(image_path, model, K, cls_names):
    im = Image.open(image_path)
    prediction_image = np.asarray(im)
    prediction_image = process_image(prediction_image)
    
    prediction = model.predict(np.expand_dims(prediction_image, axis=0))

    top_values, top_idx = tf.math.top_k(prediction, K)
    
    top_classes = [class_names[str(value + 1)] for value in top_idx.numpy()[0]]
    
    return top_values.numpy()[0], top_classes


if __name__ == "__main__":
    
    model_x = load_model(model)
    class_names = load_class_names(label_map)
    
    print(predict(image_path, model_x, K, class_names))