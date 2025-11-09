import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import numpy as np

data = pd.read_csv("./wine/wine.data", header=None)

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

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)


def build_model_one():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(13,)),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    return model


def build_model_two():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(13,), name="Input"),
            tf.keras.layers.Dense(
                16,
                activation="tanh",
                kernel_initializer="lecun_uniform",
                name="hidden_layer_one",
            ),
            tf.keras.layers.Dropout(0.3, name="dropout"),
            tf.keras.layers.Dense(
                8,
                activation="relu",
                kernel_initializer="he_normal",
                name="hiddwen_layer_two",
            ),
            tf.keras.layers.Dense(3, activation="softmax", name="Output"),
        ]
    )
    return model


def build_test_and_plot(model, X_train, X_test, y_train, y_test, epochs, batch_size):
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    history = model.fit(
        X_train, y_train, epochs=epochs, validation_split=0.2, batch_size=batch_size
    )
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    model.predict(X_test)

    plt.figure(figsize=(12, 5))

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Traning data")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()
    plt.savefig("model.png")

    return model, accuracy


model_one, accuracy_one = build_test_and_plot(
    build_model_one(), X_train, X_test, y_train, y_test, 50, 15
)
model_two, accuracy_two = build_test_and_plot(
    build_model_two(), X_train, X_test, y_train, y_test, 50, 15
)

if accuracy_one > accuracy_two:
    model_one.save("best_model.keras")
else:
    model_two.save("best_model.keras")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wine", type=float, nargs=13)
    args = parser.parse_args()

    if args.wine:
        user_input = np.array(args.wine).reshape(1, -1)
        user_input_scaled = scaler.transform(user_input)

        model = tf.keras.models.load_model("best_model.keras")
        prediction = model.predict(user_input_scaled)
        predicted_class = np.argmax(prediction, axis=1)[0] + 1

        print(f"Predicted wine class: {predicted_class}")


if __name__ == "__main__":
    main()
