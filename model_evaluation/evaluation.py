
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_METADATA = BASE_DIR / "adinkra_metadata.json"


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names(metadata_path=DEFAULT_METADATA):
    """
    Load class names from adinkra_metadata.json.

    Returns:
        dict: {class_index: class_name}
    """

    metadata_path = Path(metadata_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    class_names = {}

    for key, value in metadata.items():
        class_names[int(key)] = value["name"]

    return class_names


# ============================================================
# PLOT FULL CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    cm,
    class_names,
    save_path="confusion_matrix_101x101.png"
):
    """
    Plot and save the complete 101 x 101 confusion matrix.
    """

    num_classes = cm.shape[0]

    plt.figure(figsize=(24, 20))

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "Adinkra Symbol Confusion Matrix (101 x 101)"
    )

    plt.colorbar()

    tick_marks = np.arange(num_classes)

    labels = [
        class_names.get(i, f"Class {i}")
        for i in range(num_classes)
    ]

    plt.xticks(
        tick_marks,
        labels,
        rotation=90,
        fontsize=5
    )

    plt.yticks(
        tick_marks,
        labels,
        fontsize=5
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Full confusion matrix saved to: {save_path}"
    )


# ============================================================
# PLOT 10 x 10 CONFUSION MATRIX SECTIONS
# ============================================================

def plot_confusion_matrix_sections(
    cm,
    class_names,
    output_dir="confusion_matrix_sections"
):
    """
    Create one figure with five readable 10 x 10
    subsets from the complete confusion matrix.

    The panels show classes:
        0-9
        10-19
        20-29
        30-39
        40-49
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
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

        section = cm[start:end, start:end]

        labels = [
            class_names.get(i, f"Class {i}")
            for i in range(start, end)
        ]

        plt.sca(axis)
        plt.imshow(section, interpolation="nearest")
        plt.title(f"Classes {start}-{end - 1}")
        plt.xticks(
            np.arange(end - start),
            labels,
            rotation=45,
            ha="right",
            fontsize=7
        )
        plt.yticks(
            np.arange(end - start),
            labels,
            fontsize=7
        )

        for i in range(section.shape[0]):

            for j in range(section.shape[1]):

                value = section[i, j]

                if value != 0:

                    plt.text(
                        j,
                        i,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=7
                    )

        plt.xlabel("Predicted")
        plt.ylabel("True")

    filename = (
        output_dir /
        "confusion_matrix_5_sections.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved 5-panel confusion matrix overview: {filename}"
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    y_true,
    y_pred,
    y_prob=None,
    class_names=None,
    metadata_path=DEFAULT_METADATA,
    num_classes=101,
    val_loss=None
):
    """
    Evaluate the trained Adinkra CNN.

    Parameters
    ----------
    y_true : list
        True class labels.

    y_pred : list
        Predicted class labels.

    y_prob : array-like, optional
        Probability for every class.

        Shape:
            (number_of_samples, number_of_classes)

    class_names : dict or list, optional
        Class names.

        Dictionary format:
            {0: "Aban", 1: "Abe Dua", ...}

        List format:
            ["Aban", "Abe Dua", ...]

    metadata_path : str or Path
        Path to metadata JSON.

    num_classes : int
        Total number of model classes.

    val_loss : float, optional
        Final validation loss.

    Returns
    -------
    dict
        Evaluation results.
    """

    # ========================================================
    # LOAD CLASS NAMES
    # ========================================================

    if class_names is None:

        class_names = load_class_names(
            metadata_path
        )

    # ========================================================
    # HANDLE LIST OR DICTIONARY
    # ========================================================

    if isinstance(
        class_names,
        list
    ):

        class_names = {
            i: name
            for i, name in enumerate(
                class_names
            )
        }

    elif not isinstance(
        class_names,
        dict
    ):

        raise TypeError(
            "class_names must be either "
            "a dictionary or a list."
        )

    # ========================================================
    # CREATE TARGET NAMES
    # ========================================================

    target_names = [
        class_names.get(
            i,
            f"Class {i}"
        )
        for i in range(
            num_classes
        )
    ]

    labels = list(
        range(num_classes)
    )

    # ========================================================
    # ACCURACY
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )

    # ========================================================
    # LOG LOSS
    # ========================================================

    validation_log_loss = None

    if y_prob is not None:

        y_prob = np.asarray(
            y_prob,
            dtype=float
        )

        # ----------------------------------------------------
        # Check dimensions
        # ----------------------------------------------------

        if y_prob.ndim != 2:

            raise ValueError(
                "y_prob must be a 2D array "
                "with shape "
                "(samples, classes)."
            )

        # ----------------------------------------------------
        # Check number of classes
        # ----------------------------------------------------

        if y_prob.shape[1] != num_classes:

            raise ValueError(
                f"Expected y_prob to have "
                f"{num_classes} columns, "
                f"but got "
                f"{y_prob.shape[1]}."
            )

        # ----------------------------------------------------
        # Check probability values
        # ----------------------------------------------------

        if np.any(y_prob < 0):

            raise ValueError(
                "y_prob contains negative "
                "probability values."
            )

        # ----------------------------------------------------
        # Normalize probabilities
        # ----------------------------------------------------

        row_sums = y_prob.sum(
            axis=1,
            keepdims=True
        )

        invalid_rows = (
            row_sums.squeeze() <= 0
        )

        if np.any(
            invalid_rows
        ):

            raise ValueError(
                "Some probability rows "
                "sum to zero."
            )

        y_prob = (
            y_prob /
            row_sums
        )

        # ----------------------------------------------------
        # Calculate log loss
        # ----------------------------------------------------

        validation_log_loss = log_loss(
            y_true,
            y_prob,
            labels=labels
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n========== MODEL EVALUATION =========="
    )

    print(
        f"Number of classes: {num_classes}"
    )

    print(
        f"Accuracy: {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    if validation_log_loss is not None:

        print(
            f"Test Log Loss: "
            f"{validation_log_loss:.4f}"
        )

    if val_loss is not None:

        print(
            f"Validation Loss: "
            f"{val_loss:.4f}"
        )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\n========== CLASSIFICATION REPORT =========="
    )

    print(
        report
    )

    # ========================================================
    # SAVE FULL CONFUSION MATRIX
    # ========================================================

    full_matrix_path = (
        BASE_DIR /
        "confusion_matrix_101x101.png"
    )

    plot_confusion_matrix(
        cm,
        class_names,
        save_path=full_matrix_path
    )

    # ========================================================
    # SAVE 10 x 10 SECTIONS
    # ========================================================

    sections_dir = (
        BASE_DIR /
        "confusion_matrix_sections"
    )

    plot_confusion_matrix_sections(
        cm,
        class_names,
        output_dir=sections_dir
    )

    print(
        "\nConfusion matrix sections saved in:"
    )

    print(
        sections_dir
    )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    results = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "log_loss": validation_log_loss,
        "validation_loss": val_loss,
        "num_classes": num_classes
    }

    return results


# ============================================================
# TEST EVALUATION MODULE
# ============================================================

if __name__ == "__main__":

    print(
        "Testing evaluation module..."
    )

    example_true = [
        0,
        1,
        2,
        3
    ]

    example_pred = [
        0,
        1,
        3,
        3
    ]

    example_prob = np.zeros(
        (4, 101),
        dtype=float
    )

    example_prob[0, 0] = 1.0
    example_prob[1, 1] = 1.0
    example_prob[2, 3] = 1.0
    example_prob[3, 3] = 1.0

    results = evaluate_model(
        y_true=example_true,
        y_pred=example_pred,
        y_prob=example_prob,
        num_classes=101
    )

    print(
        "\nEvaluation module test completed."
    )
