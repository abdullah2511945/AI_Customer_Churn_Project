import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_

    indices = importances.argsort()[-10:]  # top 10

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title("Top 10 Feature Importances")
    plt.show()