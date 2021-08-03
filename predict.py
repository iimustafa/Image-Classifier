# Import the libraries
import json
import argparse
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')
 
# A R G P A S E 
parser = argparse.ArgumentParser(description = 'Image Classifier - Prediction Part')
parser.add_argument('--input', default = './test_images/hard-leaved_pocket_orchid.jpg', action="store", type=str, help='image path')
parser.add_argument('--model', default = './classifier.h5', action = "store", type = str, help='checkpoint file path/name')
parser.add_argument('--top_k', default = 5, dest = "top_k", action = "store", type = int, help='return top K most likely classes')
parser.add_argument('--category_names',dest = "category_names", action="store", default = 'label_map.json', help='mapping the categories to real names')

arg_parser = parser.parse_args()
image_path = arg_parser.input
model_path = arg_parser.model
topk = arg_parser.top_k
category_names = arg_parser.category_names


# Create the Procces Image.
def process_image(image):
    image = tf.convert_to_tensor(image, dtype = tf.float32)
    img = tf.image.resize(image, (224,224))
    img = img/255
    img = img.numpy()
    return img


# Create the Prediction Function.
def predict(image_path, model, top_k):
    img = Image.open(image_path)
    test_img = np.asarray(img)
    trandform_img = process_image(test_img)
    redim_img = np.expand_dims(trandform_img, axis = 0)
    
    prob_pred = model.predict(redim_img)
    prob_pred = prob_pred.tolist()
    probs, classes = tf.math.top_k(prob_pred, k = top_k)
    probs = probs.numpy().tolist()[0]
    classes = classes.numpy().tolist()[0]
    
    return probs, classes




if __name__ == "__main__":
    print ("start Prediction ..")
    
    with open(category_names, 'r') as f:
        class_names = json.load(f)
        
    model_2 = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    
    probs, classes = predict(image_path, model_2, topk)
    
    label_names = [class_names[str(int(idd)+1)] for idd in classes]
                                       
    print ('probs:',probs)
    print ('classes:',classes)
    print ('label_names:',label_names)
    
    print("end predection....")                        