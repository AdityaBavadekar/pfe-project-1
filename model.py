import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Downalod the model and save to saved_models/ directory.
https://drive.google.com/file/d/1-x4UlKuHSOj5KjrRdSglWi9T4cmmrj5m/view?usp=drive_link
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

class_names = """apple
avocado
bamboo
banana
barley
beans
beetroot
bell_pepper
blueberry
broccoli
cabbage
cacao
carrot
cauliflower
chickpea
chili_pepper
coconut
coffee
cotton
cottonseed
cucumber
flax
fruits
garlic
ginger
grapes
groundnut
hemp
jute
legumes
lemon
lentil
lettuce
lime
maize
mango
millet
mustard
oats
oilseeds
olives
onion
orange
papaya
pea
peach
pear
pigeon_pea
pineapple
potato
pumpkin
rice
rubber
rye
safflower
sorghum
soybean
spices
spinach
strawberry
sugarcane
sunflower
sweet_potato
tea
tobacco
tomato
turnip
watermelon
wheat""".split('\n')


class ClassificationModel:
  
  def __init__(self):
    self.model = tf.keras.models.load_model('saved_models/entire_model_2024_12_02_195243.855337.keras')
    self.img_height, self.img_width = 180, 180

  def predict(self, image_path: str):
    img = tf.keras.utils.load_img(image_path, target_size=(self.img_height, self.img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = self.model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = str(class_names[np.argmax(score)])
    prediction_confidence = 100 * np.max(score)

    return (predicted_class, prediction_confidence)