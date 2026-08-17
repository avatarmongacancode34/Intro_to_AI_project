
from model_evaluation.evaluation import evaluate_model
import matplotlib.pyplot as plt


# 1. Mock evaluation data



y_true = [0, 0, 1, 1, 2, 2]
y_pred = [0, 0, 0, 1, 2, 2]

# 2. Run model evaluation

results = evaluate_model(
    y_true=y_true,
    y_pred=y_pred
)


# 3. Display evaluation results

print("\nModel Evaluation Results")
print("------------------------")

print("\nAccuracy:")
print(results["accuracy"])

print("\nConfusion Matrix:")
print(results["confusion_matrix"])

print("\nClassification Report:")
print(results["classification_report"])

# 4. Validation Loss Curve

epochs = [1, 2, 3, 4, 5]

val_losses = [1.2, 0.95, 0.78, 0.61, 0.50]


plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    val_losses,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Curve")

plt.grid(True)
plt.tight_layout()

plt.show()
