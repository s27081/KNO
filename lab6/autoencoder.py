import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model


def prepare_data():
    train_images = tf.keras.preprocessing.image_dataset_from_directory(
        "./images_conv",
        label_mode=None,
        image_size=(128, 128),
        batch_size=5,
        color_mode="rgb",
        interpolation="bilinear",
    )
    test_images = tf.keras.preprocessing.image_dataset_from_directory(
        "./images_conv",
        label_mode=None,
        image_size=(128, 128),
        batch_size=5,
        color_mode="rgb",
        interpolation="bilinear",
    )

    def normalize(images):
        images = tf.cast(images, tf.float32) / 255.0
        return images, images

    train_images = train_images.map(normalize)
    test_images = test_images.map(normalize)

    return train_images, test_images


data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ]
)


def encoder_model(latent_dim):
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(128, 128, 3)),
            data_augmentation,
            layers.Conv2D(32, 3, activation="relu", padding="same", strides=2),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, 3, activation="relu", padding="same", strides=2),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, 3, activation="relu", padding="same", strides=2),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(latent_dim),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    return model


def decoder_model(latent_dim):
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(16 * 16 * 128, activation="relu"),
            layers.Reshape((16, 16, 128)),
            layers.Conv2DTranspose(
                128, 3, strides=2, activation="relu", padding="same"
            ),
            layers.Conv2DTranspose(64, 3, strides=2, activation="relu", padding="same"),
            layers.Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same"),
            layers.Conv2D(3, 3, activation="sigmoid", padding="same"),
        ]
    )

    return model


def autoencoder_model(encoder, decoder):
    autoencoder = Sequential([encoder, decoder], name="autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_model(model, epochs, batch_size, image_data):
    train_images, test_images = image_data

    history = model.fit(
        train_images,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(test_images),
    )

    return history


def display_images(images, autoencoder, n=5):
    for test_images, _ in images.take(1):
        decoded_imgs = autoencoder.predict(test_images)
        plt.figure(figsize=(20, 4))
        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(test_images[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()


latient_dim = 2
train_images, test_images = prepare_data()

encoder_model = encoder_model(latient_dim)
decoder_model = decoder_model(latient_dim)
autoencoder = autoencoder_model(encoder_model, decoder_model)

history = train_model(
    autoencoder, epochs=20, batch_size=5, image_data=(train_images, test_images)
)

display_images(test_images, autoencoder)
