import tensorflow as tf

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, array_to_img

model_mobile_net = load_model("models/best_mobile_net_v2.keras")
labels = []
with open("meta/labels.txt", "r") as grilled_cheese:
    lines = grilled_cheese.readlines()
    for l in lines:
        labels.append(l)


def preprocess_new_image(image_path, input_shape):
    """
    Preprocesses a new image the same way as the training data.

    Args:
    - image_path (str): Path to the image file to preprocess.
    - input_shape (tuple): Target size (height, width) for resizing the image.

    Returns:
    - np.array: Preprocessed image ready for prediction.
    """
    # Load the image
    img = load_img(image_path, target_size=input_shape)

    # Convert the image to a numpy array
    img_array = img_to_array(img)

    # Expand dimensions to match the batch shape (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image using MobileNetV2 preprocessing function
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    return img_array


def predict_image(image, input_shape=(224, 224)):
    """
    Predicts the class of a new image using the trained model.

    Args:
    - model (tf.keras.Model): The trained model for prediction.
    - image_path (str): Path to the image to predict.
    - input_shape (tuple): Target size (height, width) for resizing the image.

    Returns:
    - str: Predicted class label.
    """
    # Preprocess the image
    preprocessed_image = preprocess_new_image(image, input_shape)

    predictions = model_mobile_net.predict(preprocessed_image)

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions, axis=1)[0]

    # Get the predicted class label and its probability
    predicted_class = labels[predicted_class_index]
    predicted_probability = predictions[0][predicted_class_index]
    return predicted_class,predicted_probability

