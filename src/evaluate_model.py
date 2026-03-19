from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    roc = roc_auc_score(y_test, predictions)

    print("Accuracy:", accuracy)
    print("ROC AUC:", roc)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report")
    print(classification_report(y_test, predictions))