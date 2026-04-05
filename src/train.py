"""Train a MobileNetV2 transfer learning model for malaria cell classification.

Mirrors the notebook's model architecture, compilation, callbacks, and
training loop exactly.
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf

from preprocess import create_train_generator, create_val_generator


def build_model():
    """Build the MobileNetV2 Transfer Learning model (Method 2: CV Classification)."""
    print("Building the MobileNetV2 Transfer Learning Model...")

    IMG_SHAPE = (224, 224, 3)

    # Load the base model (MobileNetV2)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights="imagenet",
    )

    # Freeze the base model
    base_model.trainable = False

    # Custom classification head
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    # Compile the model
    learning_rate = 0.001
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    print("\nModel compiled successfully! Architecture summary:")
    model.summary()
    return model


def save_training_curves(history, output_path="models/training_curves.png"):
    """Generate and save learning curves from training history."""
    print("Generating learning curves...")

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", color="#2ca02c", linewidth=2, marker="o")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", color="#d62728", linewidth=2, marker="s")
    plt.title("Training vs. Validation Accuracy", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy Score", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", color="#1f77b4", linewidth=2, marker="o")
    plt.plot(epochs_range, val_loss, label="Validation Loss", color="#ff7f0e", linewidth=2, marker="s")
    plt.title("Training vs. Validation Loss", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss (Error)", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.show()


def train(train_dir, val_dir, model_save_path="models/malaria_cnn.h5", epochs=20):
    """Run the full training pipeline."""
    train_generator = create_train_generator(train_dir)
    val_generator = create_val_generator(val_dir)

    model = build_model()

    print("Setting up Training Callbacks...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Model Checkpointing
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    # Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )

    callbacks_list = [checkpoint, early_stopping]

    print(f"\nStarting training for up to {epochs} epochs...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks_list,
    )

    print("\nTraining Complete! The best weights have been restored and saved.")

    save_training_curves(history)
    return model, history


if __name__ == "__main__":
    train_dir = "data/processed/train"
    val_dir = "data/processed/val"
    train(train_dir, val_dir)
