
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow Python to find project files
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# PROJECT IMPORTS
# ============================================================

from adinkra_cnn import AdinkraCNN
from model_evaluation.evaluation import evaluate_model
from data_pipeline.dataset_loader import get_dataloaders


# ============================================================
# COLLECT PREDICTIONS
# ============================================================

def collect_predictions(model, data_loader, device):
    """
    Run the trained model on a dataset and collect:

    y_true = actual labels
    y_pred = predicted labels
    y_prob = probability for every class
    """

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    print("\nCollecting predictions...")

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)

            # Model outputs logits
            outputs = model(images)

            # Convert logits to probabilities
            probabilities = torch.softmax(outputs, dim=1)

            # Predicted class
            predicted = torch.argmax(
                probabilities,
                dim=1
            )

            # Store actual labels
            y_true.extend(
                labels.cpu().tolist()
            )

            # Store predictions
            y_pred.extend(
                predicted.cpu().tolist()
            )

            # Store probabilities for all 101 classes
            y_prob.extend(
                probabilities.cpu().tolist()
            )

    return y_true, y_pred, y_prob


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names,
    output_path
):
    """
    Create and save the complete 101 x 101 confusion matrix.
    """

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    plt.figure(figsize=(24, 20))

    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        "Adinkra CNN - 101 × 101 Confusion Matrix"
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    plt.xticks(
        rotation=90,
        fontsize=6
    )

    plt.yticks(
        rotation=0,
        fontsize=6
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\nFull confusion matrix saved to:\n{output_path}"
    )

    return cm


# ============================================================
# CONFUSION MATRIX 10 x 10 SECTIONS
# ============================================================

def plot_confusion_matrix_sections(
    cm,
    class_names,
    output_directory
):
    """
    Create one figure with five readable 10 x 10
    subsets from the 101 x 101 confusion matrix.

    The panels show classes:
        0-9
        10-19
        20-29
        30-39
        40-49
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    ranges = [
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 40),
        (40, 50)
    ]

    fig, axes = plt.subplots(
        1,
        len(ranges),
        figsize=(28, 6),
        constrained_layout=True
    )

    for axis, (start, end) in zip(axes, ranges):

        section_cm = cm[start:end, start:end]

        section_names = [
            class_names[i]
            if isinstance(class_names, list) and i < len(class_names)
            else class_names.get(i, f"Class {i}")
            for i in range(start, end)
        ]

        sns.heatmap(
            section_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=section_names,
            yticklabels=section_names,
            cbar=False,
            ax=axis,
            linewidths=0.3,
            linecolor="white"
        )

        axis.set_title(
            f"Classes {start}-{end - 1}"
        )

        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.tick_params(axis="x", rotation=45, labelsize=7)
        axis.tick_params(axis="y", rotation=0, labelsize=7)

    output_path = (
        output_directory /
        "confusion_matrix_5_sections.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Saved 5-panel confusion-matrix overview: "
        f"{output_path}"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # PROJECT SETTINGS
    # ========================================================

    NUM_CLASSES = 101
    BATCH_SIZE = 16

    # ========================================================
    # MODEL PATH
    # ========================================================

    best_model_path = (
        PROJECT_ROOT /
        "best_adinkra_model.pth"
    )

    if not best_model_path.exists():

        raise FileNotFoundError(
            f"Could not find model:\n"
            f"{best_model_path}"
        )

    print(
        "\nExisting trained model found:"
    )

    print(
        best_model_path
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print(
        "\nLoading dataset..."
    )

    train_loader, val_loader, test_loader = (
        get_dataloaders(
            batch_size=BATCH_SIZE
        )
    )

    print(
        "\nDataset loaded:"
    )

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print(
        "\nCreating Adinkra CNN..."
    )

    model = AdinkraCNN(
        num_classes=NUM_CLASSES
    ).to(device)

    print(
        f"Model created with "
        f"{NUM_CLASSES} output classes."
    )

    # ========================================================
    # LOAD EXISTING TRAINED WEIGHTS
    # ========================================================

    print(
        "\nLoading existing trained model..."
    )

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device
        )
    )

    model.eval()

    print(
        "Best model loaded successfully."
    )

    # ========================================================
    # GET CLASS NAMES
    # ========================================================

    class_names = test_loader.dataset.dataset.classes

    print(
        f"\nNumber of dataset classes: "
        f"{len(class_names)}"
    )

    # ========================================================
    # COLLECT TEST PREDICTIONS
    # ========================================================

    y_true, y_pred, y_prob = (
        collect_predictions(
            model,
            test_loader,
            device
        )
    )

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print(
        "\n========== FINAL TEST EVALUATION =========="
    )

    # We do NOT pass val_losses here because this script
    # is evaluating the already-trained model.
    results = evaluate_model(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n========== FINAL RESULTS =========="
    )

    print(
        f"Test Accuracy: "
        f"{results['accuracy']:.4f}"
    )

    if results.get("validation_loss") is not None:

        print(
            f"Test Log Loss: "
            f"{results['validation_loss']:.4f}"
        )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    confusion_directory = (
        PROJECT_ROOT /
        "model_evaluation" /
        "confusion_matrices"
    )

    confusion_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    full_cm_path = (
        confusion_directory /
        "confusion_matrix_101x101.png"
    )

    cm = plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_path=full_cm_path
    )

    # ========================================================
    # FIVE 10 x 10 READABLE SECTIONS
    # ========================================================

    print(
        "\nCreating readable 10 x 10 confusion-matrix sections..."
    )

    plot_confusion_matrix_sections(
        cm=cm,
        class_names=class_names,
        output_directory=confusion_directory
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print(
        "\n============================================"
    )

    print(
        "MODEL EVALUATION COMPLETED SUCCESSFULLY"
    )

    print(
        "============================================"
    )

    print(
        f"\nConfusion matrices are saved in:"
    )

    print(
        confusion_directory
    )
