import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import keras_tuner as kt
import argparse
from sklearn.metrics import confusion_matrix

def prepare_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = np.expand_dims(train_images, -1)
    test_images = np.expand_dims(test_images, -1)

    return train_images, train_labels, test_images, test_labels


def build_dense_model(hp=None):
    if hp:
        hp_units = hp.Choice("units", [8, 16, 32, 64])
        learning_rate = hp.Float("lr", 1e-3, 1e-2, sampling="log")
        dropout_rate = hp.Float("dropout", 0.0, 0.6, step=0.1)
    else:
        hp_units = 32
        learning_rate = 0.001
        dropout_rate = 0.2

    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(28, 28,1)),
            keras.layers.Dense(units=hp_units, activation="relu"),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(10, activation="softmax"),
        ],
        name="dense"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_conv_model(hp=None):
    if hp:
        hp_filters = hp.Choice('filters', [32, 64, 96])
        learning_rate = hp.Float("lr", 1e-3, 1e-2, sampling="log")
    else:
        hp_filters = 32
        learning_rate = 0.001

    model = keras.Sequential(
        [
            keras.layers.Conv2D(hp_filters, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ],
        name="conv"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def test_evaluate_and_save(model, epochs, batch_size, image_data):
    train_images, train_labels, test_images, test_labels = image_data

    history = model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(test_images, test_labels),
    )

    loss, accuracy = model.evaluate(test_images, test_labels)
    y_pred = np.argmax(model.predict(test_images), axis=1)
    cm = confusion_matrix(test_labels, y_pred)

    model.save(f"{model.name}_model.keras")

    with open(f"results_{model.name}.txt", "w") as f:
        print("Model Validation Accuracy:", accuracy, file=f)
        print("Model Validation Loss:", loss, file=f)
        print("\nConfusion matrix:", file=f)
        print(cm, file=f)

    return model, accuracy, history, loss

def use_tuner(model, model_name, image_data, epochs=5, batch_size=32):
    train_images, train_labels, test_images, test_labels = image_data
    tuner = kt.RandomSearch(
                model,
                objective='val_accuracy',
                max_trials=1,
                directory='tuning_results',
                project_name=model_name
        )
    
    tuner.search(
            train_images, 
            train_labels, 
            epochs=epochs, 
            validation_data=(test_images, test_labels)
        )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]
        
    print("\n--- Najlepsze Hiperparametry ---")
    print(best_hps.values)
    test_evaluate_and_save(best_model, epochs, batch_size, image_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuner", "-t", type=bool, default=False)
    args = parser.parse_args()
    is_tuner = args.tuner

    image_data = prepare_data()
    if is_tuner:
        use_tuner(build_conv_model,"conv_model", image_data)
        use_tuner(build_dense_model, "dense_model", image_data)
    else:
        dense_model = build_dense_model()
        conv_model = build_conv_model()
        test_evaluate_and_save(dense_model, epochs=5, batch_size=32, image_data=image_data)

        test_evaluate_and_save(conv_model, epochs=5, batch_size=32, image_data=image_data)
if __name__ == "__main__":
    main()
