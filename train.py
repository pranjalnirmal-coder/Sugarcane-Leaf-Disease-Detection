"""
Train a CNN using MobileNetV2 transfer learning and generate
an evaluation report for sugarcane disease classification.

Dataset structure:
dataset/
    Healthy/
        image1.jpg
        image2.jpg
    Red Rot/
        image1.jpg
    Smut/
        image1.jpg
    Brown Rust/
        image1.jpg

Run:
    python train.py --data dataset --epochs 10
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix


# --------------------------------------------------
# 1. Parse command-line arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Train a MobileNetV2 sugarcane disease classifier"
)

parser.add_argument(
    "--data",
    type=str,
    default="dataset",
    help="Path to the dataset directory"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    help="Number of initial training epochs"
)

args = parser.parse_args()


# --------------------------------------------------
# 2. Create output directory
# --------------------------------------------------

os.makedirs("models", exist_ok=True)


# --------------------------------------------------
# 3. Load training and validation datasets
# --------------------------------------------------

dataset_args = {
    "directory": args.data,
    "image_size": (224, 224),
    "batch_size": 32,
    "validation_split": 0.2,
    "seed": 42,
    "label_mode": "int",
}

train_dataset = tf.keras.utils.image_dataset_from_directory(
    subset="training",
    shuffle=True,
    **dataset_args
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    subset="validation",
    shuffle=False,
    **dataset_args
)

class_names = train_dataset.class_names
num_classes = len(class_names)

print("\nDetected classes:", class_names)
print("Number of classes:", num_classes)


# Save class names
with open("models/classes.json", "w") as file:
    json.dump(class_names, file, indent=4)


# --------------------------------------------------
# 4. Image augmentation
# --------------------------------------------------

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.15),
])


# --------------------------------------------------
# 5. Image preprocessing
# --------------------------------------------------

def preprocess(images, labels):
    """
    Normalize pixel values from [0, 255] to [-1, 1].
    This matches MobileNetV2 preprocessing.
    """
    images = tf.cast(images, tf.float32)
    images = images / 127.5 - 1.0

    return images, labels


def augment_and_preprocess(images, labels):
    images = augmentation(images, training=True)
    return preprocess(images, labels)


train_dataset = train_dataset.map(
    augment_and_preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

val_dataset = val_dataset.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------
# 6. Load MobileNetV2 pretrained model
# --------------------------------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers during initial training
base_model.trainable = False


# --------------------------------------------------
# 7. Build the CNN classification model
# --------------------------------------------------

inputs = tf.keras.Input(shape=(224, 224, 3))

features = base_model(inputs, training=False)

features = tf.keras.layers.GlobalAveragePooling2D()(features)

features = tf.keras.layers.Dropout(0.3)(features)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax",
    name="predictions"
)(features)

model = tf.keras.Model(inputs, outputs)

model.summary()


# --------------------------------------------------
# 8. Initial training
# --------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\nStarting initial training...")

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=args.epochs,
    callbacks=[early_stopping]
)


# --------------------------------------------------
# 9. Fine-tune the last MobileNetV2 layers
# --------------------------------------------------

base_model.trainable = True

# Freeze all but the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen for stability
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nStarting fine-tuning...")

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=max(3, args.epochs // 2),
    callbacks=[early_stopping]
)


# --------------------------------------------------
# 10. Save the trained model
# --------------------------------------------------

model.save("models/sugarcane_cnn.keras")

print("\nModel saved to models/sugarcane_cnn.keras")


# --------------------------------------------------
# 11. Evaluate the model
# --------------------------------------------------

print("\nGenerating evaluation report...")

true_labels = np.concatenate([
    labels.numpy()
    for _, labels in val_dataset
])

predictions = model.predict(val_dataset)

predicted_labels = np.argmax(predictions, axis=1)


# Classification report
report = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(num_classes)),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

# Confusion matrix
matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(num_classes))
)

report["confusion_matrix"] = matrix.tolist()


# --------------------------------------------------
# 12. Save evaluation results
# --------------------------------------------------

with open("models/evaluation.json", "w") as file:
    json.dump(report, file, indent=4)

print("\nClassification Report:\n")

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    )
)

print("\nConfusion Matrix:\n")
print(matrix)

print("\nEvaluation report saved to models/evaluation.json")
print("\nTraining and evaluation completed successfully!")


