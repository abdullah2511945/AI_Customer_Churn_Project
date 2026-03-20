from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def plot_roc_curve(model, X_test, y_test):

    probs = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probs)

    roc_auc = auc(fpr, tpr)

    plt.figure()

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")
    plt.legend()

    plt.show()

def plot_model_performance(results_df):

    results_df.set_index("Model").plot(kind="bar")

    plt.title("Model Performance Comparison")
    plt.ylabel("Score")

    plt.xticks(rotation=0)
    plt.legend()

    plt.show()