import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from preprocess import create_test_generator, IMG_SIZE, BATCH_SIZE
from image_utils import apply_gaussian_blur


def evaluate_model(model_path="models/malaria_cnn.h5", test_dir="data/processed/test"):
    """Load the best model and evaluate on the unseen test set."""

    print("Loading the best model for final evaluation...")
    best_model = tf.keras.models.load_model(model_path)

    test_generator = create_test_generator(test_dir)

    print("\nRunning predictions on the Test Set. This might take a minute...")
    predictions = best_model.predict(test_generator)

    # Convert probabilities to binary labels using a 0.5 threshold
    y_pred = (predictions > 0.5).astype(int).reshape(-1)

    # Get the true labels from the generator
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    # Classification Report
    print("\n" + "=" * 40)
    print("--- CLASSIFICATION REPORT ---")
    print("=" * 40)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 14},
    )
    plt.title("Confusion Matrix - Unseen Test Set", fontsize=16)
    plt.ylabel("True Real-World Label", fontsize=12)
    plt.xlabel("Model Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=100, bbox_inches="tight")
    plt.show()

    # ROC Curve
    y_score = predictions.reshape(-1)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    print(f"\nAUC Score: {roc_auc:.4f}")

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", linewidth=1, label="Random Baseline")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Unseen Test Set")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=100, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Resolve model path (check multiple candidates)
    model_candidates = [
        "models/best_model.h5",
        "models/malaria_cnn.h5",
        "../models/best_model.h5",
        "../models/malaria_cnn.h5",
    ]

    resolved_model_path = next((p for p in model_candidates if os.path.exists(p)), None)
    if resolved_model_path is None:
        raise FileNotFoundError(
            "Could not find a model file. Checked: " + ", ".join(model_candidates)
        )

    # Resolve test directory
    test_candidates = [
        "data/processed/test",
        "../data/processed/test",
    ]

    resolved_test_dir = next((p for p in test_candidates if os.path.isdir(p)), None)
    if resolved_test_dir is None:
        raise FileNotFoundError(
            "Could not find test directory. Checked: " + ", ".join(test_candidates)
        )

    evaluate_model(resolved_model_path, resolved_test_dir)
