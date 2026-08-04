from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    print("Model Accuracy:", accuracy)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


# Example placeholder
# y_true = actual labels
# y_pred = model predictions
