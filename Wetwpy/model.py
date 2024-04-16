import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained ResNet50 model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Function to process the image
def process_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Function to predict dog breeds based on the image
def predict_breed(img_path):
    processed_img = process_image(img_path)
    preds = model.predict(processed_img)
    decoded_preds = decode_predictions(preds, top=3)[0]  # Select the top 3 most likely predictions
    predictions = []
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        predictions.append(f"{i + 1}. {label}: {score:.2f}")
    return predictions
