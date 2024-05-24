from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(y_true, y_pred):
    unique_labels = sorted(list(set(y_true) | set(y_pred)))  # Determine unique classes in y_true and y_pred
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=unique_labels))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=unique_labels))
