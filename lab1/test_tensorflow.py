import os.path

import keras.models
import numpy as np
import tensorflow as tf
import argparse
from keras.preprocessing import image
import matplotlib.pyplot as plt

IMG_WIDTH, IMG_HEIGHT = 28, 28

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str)
args = parser.parse_args()
image_path = args.image

if not os.path.isfile("tutorial-model.keras"):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)
    model.save("tutorial-model.keras")
else:
    model = keras.models.load_model("tutorial-model.keras")

img_load = image.load_img(
    image_path, color_mode="grayscale", target_size=(IMG_WIDTH, IMG_HEIGHT)
)
img_tensor = image.img_to_array(img_load) / 255.0
img = np.array([img_tensor])
prediction = model.predict(img)

print(prediction[0])
