import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Downalod the model and save to saved_models/ directory.
https://drive.google.com/file/d/1-x4UlKuHSOj5KjrRdSglWi9T4cmmrj5m/view?usp=drive_link
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

class ClassificationModel:
  
  def __init__(self):
    """
    Initialize the ClassificationModel class.
    - Loads the pre-trained model from the specified path.
    - Sets image dimensions for resizing during prediction.
    """
    IMAGE_DIMS = (180, 180)
    self.model = tf.keras.models.load_model('saved_models/model_v1.keras')
    self.img_height, self.img_width = IMAGE_DIMS
    self.load_class_names()
    
  def load_class_names(self):
    """
    Load the class names from the classes.json file.
    """
    with open('classes.json') as f:
      self.CLASS_NAMES = json.load(f)

  def predict(self, image_path: str):
    """
    Predict the class of an image given its file path.
    Returns:
        Tuple[str, float]: Predicted class label and confidence score as a percentage.
    """
    img = tf.keras.utils.load_img(image_path, target_size=(self.img_height, self.img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = self.model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = str(self.CLASS_NAMES[np.argmax(score)])
    prediction_confidence = 100 * np.max(score)

    return (predicted_class, prediction_confidence)