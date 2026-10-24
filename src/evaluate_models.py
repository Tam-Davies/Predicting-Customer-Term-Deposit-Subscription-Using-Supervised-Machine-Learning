import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


class ModelEvaluator:
    """Evaluate supervised classification models."""

    def __init__(self, models):
        self.models = models

    def evaluate(self, X_test, y_test):
        """Evaluate all models using classification metrics."""

        results = []

        for name, model in self.models.items():

            predictions = model.predict(X_test)

            probabilities = model.predict_proba(X_test)[:, 1]

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(
                    y_test,
                    predictions
                ),
                "Precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),
                "Recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),
                "ROC-AUC": roc_auc_score(
                    y_test,
                    probabilities
                )
            })

        return pd.DataFrame(results)

    def confusion_matrices(self, X_test, y_test):
        """Return confusion matrices for all models."""

        matrices = {}

        for name, model in self.models.items():

            predictions = model.predict(X_test)

            matrices[name] = confusion_matrix(
                y_test,
                predictions
            )

        return matrices