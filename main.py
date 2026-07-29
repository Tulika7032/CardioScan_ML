import os
import time
import logging
import pandas as pd

from src.preprocess import load_data, preprocess_data, split_data
from src.train import train_model, save_model
from src.evaluate import evaluate_model

# ------------------------------------------------------------------
# Configure Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------------------------------------------------------
# Create Output Directories
# ------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def main():
    try:

        # ----------------------------------------------------------
        # Load Dataset
        # ----------------------------------------------------------
        logging.info("Loading dataset...")

        df = load_data("data/heart-disease.csv")

        logging.info(f"Dataset loaded successfully. Shape: {df.shape}")

        # ----------------------------------------------------------
        # Preprocessing
        # ----------------------------------------------------------
        logging.info("Preprocessing data...")

        X, y = preprocess_data(df)

        logging.info(f"Features: {X.shape}")
        logging.info(f"Target: {y.shape}")

        # ----------------------------------------------------------
        # Train-Test Split
        # ----------------------------------------------------------
        logging.info("Splitting dataset...")

        X_train, X_test, y_train, y_test = split_data(X, y)

        logging.info(
            f"Train: {X_train.shape}, Test: {X_test.shape}"
        )

        # ----------------------------------------------------------
        # Model Training
        # ----------------------------------------------------------
        logging.info("Training models...")

        start = time.time()

        models = train_model(X_train, y_train)

        end = time.time()

        logging.info(
            f"Training completed in {end-start:.2f} seconds"
        )

        # ----------------------------------------------------------
        # Evaluation
        # ----------------------------------------------------------
        results = {}
        summary = []

        for name, model in models.items():

            print("\n" + "=" * 60)
            print(f"Evaluating {name.upper()}")
            print("=" * 60)

            metrics = evaluate_model(
                model,
                X_test,
                y_test
            )

            results[name] = metrics

            summary.append({
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1"],
                "ROC-AUC": metrics["roc_auc"]
            })

        # ----------------------------------------------------------
        # Save Model Comparison CSV
        # ----------------------------------------------------------
        comparison_df = pd.DataFrame(summary)

        comparison_df.to_csv(
            "reports/model_comparison.csv",
            index=False
        )

        logging.info(
            "Saved model comparison report."
        )

        # ----------------------------------------------------------
        # Select Best Model
        # ----------------------------------------------------------
        best_model_name = max(
            results,
            key=lambda x: results[x]["f1"]
        )

        best_model = models[best_model_name]

        print("\n" + "=" * 60)
        print(f"BEST MODEL : {best_model_name.upper()}")
        print("=" * 60)

        for metric, value in results[best_model_name].items():

            if metric not in [
                "confusion_matrix",
                "classification_report"
            ]:
                print(f"{metric:<12}: {value:.4f}")

        # ----------------------------------------------------------
        # Save Best Model
        # ----------------------------------------------------------
        save_model(best_model)

        logging.info(
            "Best model saved successfully."
        )

        # ----------------------------------------------------------
        # Save Metrics Report
        # ----------------------------------------------------------
        with open(
            "reports/model_metrics.txt",
            "w"
        ) as file:

            for model_name, metrics in results.items():

                file.write("=" * 50 + "\n")
                file.write(f"{model_name.upper()}\n")
                file.write("=" * 50 + "\n")

                for metric, value in metrics.items():

                    if metric not in [
                        "confusion_matrix",
                        "classification_report"
                    ]:
                        file.write(
                            f"{metric}: {value}\n"
                        )

                file.write("\n")

        logging.info(
            "Metrics report generated."
        )

        # ----------------------------------------------------------
        # Save Best Model Name
        # ----------------------------------------------------------
        with open(
            "reports/best_model.txt",
            "w"
        ) as file:

            file.write(best_model_name)

        logging.info(
            "Best model report saved."
        )

        print("\nProject completed successfully!")

    except Exception as e:

        logging.exception(
            "Pipeline execution failed."
        )

        print(f"\nError: {e}")


if __name__ == "__main__":
    main()