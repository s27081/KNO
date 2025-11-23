import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import argparse
import keras_tuner as kt
from sklearn.metrics import confusion_matrix
import numpy as np

"""
Lab 3 prediction:
- model_one: Validation Loss: 0.1541, Accuracy: 0.9722
- model_two: Validation Loss: 0.1463, Accuracy: 0.9444
"""

data = pd.read_csv("../lab3/wine/wine.data", header=None)

columns = [
    "Class",
    "Alcohol",
    "Malic_Acid",
    "Ash",
    "Alcalinity_of_Ash",
    "Magnesium",
    "Total_Phenols",
    "Flavanoids",
    "Nonflavanoid_Phenols",
    "Proanthocyanins",
    "Color_Intensity",
    "Hue",
    "OD280/OD315",
    "Proline",
]

data.columns = columns

data = data.sample(frac=1, random_state=8)

X = data.drop("Class", axis=1).values
y = data["Class"].values - 1

y = tf.keras.utils.to_categorical(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

data_normalizer = tf.keras.layers.Normalization()
data_normalizer.adapt(X_train)


def build_base_model():
    model = tf.keras.Sequential(
        [
            data_normalizer,
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dropout(0.3, name="dropout"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    return model


def build_hp_model(hp):
    model = tf.keras.Sequential(
        [
            data_normalizer,
            tf.keras.layers.Dense(hp.Choice("units", [8, 16, 32]), activation="relu"),
            tf.keras.layers.Dropout(hp.Float("dropout", 0.0, 0.5, step=0.1)),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    learninig_rate = hp.Float("lr", 1e-4, 1e-2, sampling="log")

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learninig_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def build_and_test(model, X_train, X_test, y_train, y_test, epochs, batch_size):
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size
    )
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    model.predict(X_test)

    return model, accuracy, history


base_model, accuracy, history = build_and_test(
   build_base_model(), X_train, X_test, y_train, y_test, 50, 15
)

base_accuracy = history.history["val_accuracy"][-1]

base_model.save("base_model.keras")

tuner = kt.RandomSearch(
    build_hp_model,
    objective="val_loss",
    max_trials=5,
    project_name="wine_prediction")

tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
best_model = tuner.get_best_models()[0]
best_hyperparameters = tuner.get_best_hyperparameters()[0]

best_model.save("best_hp_model.keras")

y_pred = np.argmax(best_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)

print("========")
print("Base Model Validation Accuracy:", base_accuracy)
print("Hyperparameters Model Validation Accuracy", best_model)
print("Hyperparameters: ", best_hyperparameters.values)
print("\nConfusion matrix:")
print(cm)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wine", type=float, nargs=13)
    args = parser.parse_args()

    if args.wine:
        user_input = np.array(args.wine).reshape(1, -1)

        model = tf.keras.models.load_model("model.keras")
        prediction = model.predict(user_input)
        predicted_class = np.argmax(prediction, axis=1)[0] + 1

        print(f"Predicted wine class: {predicted_class}")


if __name__ == "__main__":
    main()
