import pandas as pd
import matplotlib.pyplot as plt


def plot_feature_importance(model, feature_names):

    importance = model.feature_importances_

    features = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    features = features.sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10,6))
    plt.barh(features["Feature"], features["Importance"])
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()