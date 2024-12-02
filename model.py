import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

model = tf.keras.models.load_model('saved_models/entire_model_2024_12_02_195243.855337.keras')

img_height, img_width = 180, 180
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

# model.summary()

def pred(crop_img_url:str):
  crop_img_path = tf.keras.utils.get_file(origin=crop_img_url)

  img = tf.keras.utils.load_img(
      crop_img_path, target_size=(img_height, img_width)
  )
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )
  
# Cotton Image
image_url = 'https://www.cottonusa.org/uploads/images/cottonball.jpg'

pred(image_url)