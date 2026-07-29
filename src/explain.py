import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(model, feature_names):

    rf = model.named_steps["classifier"]

    importance = rf.feature_importances_

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    df = df.sort_values("Importance", ascending=False)

    plt.figure(figsize=(8,5))
    plt.barh(df["Feature"], df["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("reports/feature_importance.png")