import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)


# Always locate metadata file inside this folder
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA = BASE_DIR / "adinkra_metadata.json"


def load_class_names(metadata_path=DEFAULT_METADATA):
    metadata_file = Path(metadata_path)

    if not metadata_file.exists():
        return {}

    with metadata_file.open("r", encoding="utf-8") as file_handle:
        metadata = json.load(file_handle)

    class_names = {}

    for class_id, values in metadata.items():
        if isinstance(values, dict):
            name = values.get("name", "")
        else:
            name = ""

        class_names[str(class_id)] = name or f"Class {class_id}"

    return class_names

def create_evaluation_visualizations(confusion, labels, val_losses=None):
    if val_losses:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(range(1, len(val_losses) + 1), val_losses, marker="o")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Validation Loss")
        axes[0].set_title("Validation Loss Curve")
        axes[0].grid(True, alpha=0.3)

        sns.heatmap(
            confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[1],
        )
        axes[1].set_xlabel("Predicted Label")
        axes[1].set_ylabel("True Label")
        axes[1].set_title("Confusion Matrix Heatmap")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            confusion,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix Heatmap")

    plt.tight_layout()
    plt.show()


def evaluate_model(
    y_true,
    y_pred,
    y_prob=None,
    val_losses=None,
    class_names=None,
    metadata_path=DEFAULT_METADATA,
):

    # Safety check
    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same number of samples."
        )

    # Load class names if not provided
    if class_names is None:
        class_names = load_class_names(metadata_path)

    # Get all labels that appear
    labels = sorted(set(y_true) | set(y_pred))

    # Convert IDs into class names
    target_names = [
        class_names.get(str(label), f"Class {label}")
        for label in labels
    ]

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Confusion Matrix
    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    # Classification Report
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    # Validation Loss (optional)
    validation_loss = None

    if y_prob is not None:
        validation_loss = log_loss(
            y_true,
            y_prob,
            labels=labels
        )


    # Display results
    print("Model Accuracy:", accuracy)

    print("\nValidation Loss:")
    if validation_loss is None:
        print(
            "Unavailable - pass y_prob with class probabilities "
            "to compute log loss."
        )
    else:
        print(validation_loss)

    print("\nConfusion Matrix:")
    print(confusion)
    create_evaluation_visualizations(confusion, target_names, val_losses=val_losses)

    print("\nClassification Report:")
    print(report)


    return {
        "accuracy": accuracy,
        "validation_loss": validation_loss,
        "confusion_matrix": confusion,
        "classification_report": report,
        "labels": labels,
        "target_names": target_names,
    }
# Optional test block
# Remove this before final submission if your team does not want test code here
if __name__ == "__main__":

    y_true = [0, 1, 2, 1, 0, 2]
    y_pred = [0, 1, 2, 0, 0, 2]
    val_losses = [1.2, 0.95, 0.78, 0.61, 0.5]

    results = evaluate_model(
        y_true,
        y_pred,
        val_losses=val_losses
    )


